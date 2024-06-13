import json

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from model.pre_model import tokenizer
import os


class LeafDataSet(Dataset):
    def __init__(self, annotation_path, image_path="./", crop_size=224, block_size=128, start_eq_end=True):
        self.annotations = []
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.start_eq_end = start_eq_end
        with open(f"{annotation_path}", "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        self.data_size = len(self.annotations)
        self.block_size = block_size
        print("All data size", self.data_size)
        print("", len(self.tokenizer.get_vocab().keys()))

    def __getitem__(self, id):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        block_size = self.block_size
        annotation = self.annotations[id]

        image = annotation["image"]
        img_id = annotation["image_id"]
        image = Image.open(f"{self.image_path}/{image}").convert('RGB')
        image = self.transform(images=image, return_tensors="pt")
        image = image["pixel_values"]

        text = annotation["text"]
        text = self.tokenizer(text).input_ids

        text = text[:block_size + 2]
        text = [bos_token_id] + text
        text[-1] = eos_token_id
        text = torch.Tensor(text)

        caption = annotation["caption"]
        caption = self.tokenizer(caption).input_ids
        caption = caption[:block_size + 2]
        caption = [bos_token_id] + caption
        caption[-1] = eos_token_id
        caption = torch.Tensor(caption)

        return img_id, image, text, caption

    def __len__(self):
        return self.data_size


class LeafTemplateDataSet(Dataset):
    def __init__(self, annotation_path, image_path="./", crop_size=224, block_size=128,
                 start_eq_end=True, not_template=False, debug=False):
        self.annotations = []
        self.templates = []
        self.feature_dict = {}
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.start_eq_end = start_eq_end
        self.debug = debug
        with open(f"{annotation_path}", "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        base_path = os.path.dirname(annotation_path)
        with open(f"{base_path}/template.json", "r", encoding="utf-8") as f:
            self.templates = json.load(f)
        if not_template:
            self.templates = self.templates[0:1]
        with open(f"{base_path}/leaf_feature_dict.json", "r", encoding="utf-8") as f:
            self.feature_dict = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            # transforms.Resale(1.0 / 255.0, 0),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        self.annotation_size = len(self.annotations)
        self.template_size = len(self.templates)
        self.data_size = self.template_size * self.annotation_size
        self.block_size = block_size
        print("All data size", self.data_size)
        print("", len(self.tokenizer.get_vocab().keys()))

        feature_dict_token = {}
        feature_key_token = {}
        for k, v in self.feature_dict.items():
            feature_key_token[k] = torch.tensor(self.tokenizer(f"{k}").input_ids, dtype=torch.long)
            feature_dict_token[k] = {}
            for kk, vv in v.items():
                feature_dict_token[k][kk] = torch.tensor(self.tokenizer(" " + vv).input_ids, dtype=torch.long)
        self.feature_key_token, self.feature_dict_token = feature_key_token, feature_dict_token
        self.start_instruct_tokens = torch.tensor(self.tokenizer("{ {").input_ids, dtype=torch.long)
        self.end_instruct_tokens = torch.tensor(self.tokenizer("} }. },").input_ids, dtype=torch.long)

    def __getitem__(self, id):
        annotation = self.annotations[int(id / self.template_size)]
        image = annotation["image"]
        img_id = annotation["image_id"]

        if self.debug:
            try:
                image=Image.open(f"{self.image_path}/{image}").convert('RGB')
                image = self.transform(image)
            except:
                print("NOT transform", img_id)
            return 0, img_id, torch.randn((3, 244, 244)), torch.ones(5), torch.ones(5)
        
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        block_size = self.block_size

        template_id = id - int(id / self.template_size) * self.template_size
        template = self.templates[template_id]

        image = Image.open(f"{self.image_path}/{image}").convert('RGB')
        image = self.transform(image)

        text = template
        text = self.tokenizer(text).input_ids

        text = text[:block_size + 1]
        text = [bos_token_id] + text
        text = text + [eos_token_id]
        text = torch.Tensor(text)
        label = {"color": self.feature_dict['color'][str(annotation['color'])],
                 "texture": self.feature_dict['texture'][str(annotation['texture'])],
                 "morphology": self.feature_dict['morphology'][str(annotation['morphology'])],
                 "situated": self.feature_dict['situated'][str(annotation['situated'])],
                 "area": self.feature_dict['area'][str(annotation['area'])],
                 "address": self.feature_dict['address'][str(annotation['address'])]}
        caption = template.format(**label)
        caption = self.tokenizer(caption).input_ids
        caption = caption[:block_size + 1]
        caption = [bos_token_id] + caption
        caption[-1] = eos_token_id
        caption = torch.Tensor(caption)

        return template_id, img_id, image, text, caption

    def __len__(self):
        return self.data_size

    def __getattr__(self, item):
        if item == "feature_key_dict_token":
            return self.feature_key_token, self.feature_dict_token


def collate_fn(data):
    data.sort(key=lambda x: len(x[4]), reverse=True)
    template_ids, img_ids, images, texts, captions = zip(*data)  # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    img_ids = list(img_ids)
    # texts = torch.stack(texts, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(txt) for txt in texts]
    text1 = torch.zeros(len(texts), max(text_lengths)).long().fill_(50254)
    text2 = torch.zeros(len(texts), max(text_lengths)).long()

    target_lengths = [len(cap) for cap in captions]
    target1 = torch.zeros(len(captions), max(target_lengths)).long().fill_(50254)
    target2 = torch.zeros(len(captions), max(target_lengths)).long()

    for i, txt in enumerate(texts):
        end = text_lengths[i]
        text1[i, :end] = txt[:end]
        text2[i, :end] = txt[:end]

    for i, cap in enumerate(captions):
        end = target_lengths[i]
        target1[i, :end] = cap[:end]
        target2[i, :end] = cap[:end]
    return images, text1, target1, text2, target2, text_lengths, target_lengths, img_ids, template_ids


if __name__ == '__main__':
    # from tqdm import tqdm
    #
    # leaf = LeafTemplateDataSet("annotation6/leaf_caption_train.json", "/home2/zhaokejun/datasets/leaf-label",
    #                            start_eq_end=False)
    #
    # print(leaf.feature_dict)
    # print(leaf.feature_dict_token)
    # pass
    # annotation="annotation6"
    # try:
    #     if os.getlogin() == "zhaokejun":
    #         leaf = LeafTemplateDataSet(f"{annotation}/leaf_caption_train.json",
    #                                    "/home2/zhaokejun/datasets/leaf-label",
    #                                    start_eq_end=False)
    #     else:
    #         leaf = LeafTemplateDataSet(f"{annotation}/leaf_caption_debug.json", "images", start_eq_end=False)
    # except:
    #     leaf = LeafTemplateDataSet(f"{annotation}/leaf_caption_train.json", "/home2/zhaokejun/datasets/leaf-label",
    #                                start_eq_end=False)
    # data_loader = DataLoader(leaf, batch_size=1, collate_fn=collate_fn)
    # for idx, (images, texts, targets, text2, target2, text_lengths, target_lengths, img_ids) in enumerate(data_loader):
    #     print(img_ids)
    image = Image.open(
        "/home2/zhaokejun/datasets/leaf-label/apple_round_spot_disease/908fa0ec08fa513d2697c24a402042fbb2fb4316735b.txt").convert(
        'RGB')
