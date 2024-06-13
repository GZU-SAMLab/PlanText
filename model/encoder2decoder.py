import copy
import os.path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model.encoder import Encoder
from model.decode import Decoder
import numpy as np


class Encoder2Decoder(nn.Module):

    def __init__(self, hidden_size, vocab_size=50257, max_length=128, device=None):
        super(Encoder2Decoder, self).__init__()
        self.encoder = Encoder(hidden_size=hidden_size, d_model=768)
        self.decoder = Decoder(hidden_size, vocab_size)
        self.max_length = max_length
        self.device = device

    def forward(self, image, text=None, caption=None, lengths=None):
        if caption is not None:
            if torch.cuda.device_count() > 1:
                device_ids = range(torch.cuda.device_count())
                encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
                T, features, (_, _) = encoder_parallel(image, text)
            else:
                T, features, (_, _) = self.encoder(image, text)
            return self.output(T, features, caption, lengths)

    # 创建一个逆标准化函数
    def denormalize(self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # Clone tensor to avoid modifying the input tensor
        tensor = tensor.clone()
        tensor = tensor.cpu()

        # Convert tensor to numpy array
        tensor = tensor.numpy().transpose(1, 2, 0)

        # Denormalize each channel
        for i in range(3):
            tensor[..., i] = tensor[..., i] * std[i] + mean[i]

        # Clip values to range [0, 1] in case of numerical errors
        tensor = np.clip(tensor, 0, 1)

        return tensor

    def output(self, T, features, caption, lengths=None, states=None, image=None, text=None):

        if lengths == None:
            lengths = torch.ones(caption.size(0)).fill_(caption.size(1))
        scores, states, atten_weights, beta = self.decoder(T, features, caption, states)
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)
        return packed_scores

    def generate(self, image: torch.Tensor, text, max_len=128, common=10):
        device = self.device
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
            T, features, (image_features, text_features) = encoder_parallel(image, text)
        else:
            T, features, (image_features, text_features) = self.encoder(image, text)

        caption = torch.LongTensor(image.size(0), 1).fill_(50255).to(device)
        lengths = torch.ones(image.size(0))
        results = torch.LongTensor(image.size(0), max_len + 2).fill_(50255).to(device)
        with torch.no_grad():
            for i in range(max_len):
                result = self.output(T, features, caption, lengths)
                result: torch.Tensor
                result = result[0]
                result = result.unsqueeze(1)

                results[:, i:i + 1] = result.argmax(dim=2)
                caption = result.argmax(dim=2)

        return results

    def generate_reinforcement_reset(self, image: torch.Tensor, text):
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
            T, features, (_, _) = encoder_parallel(image, text)
        else:
            T, features, (_, _) = self.encoder(image, text)
        self.T = T
        self.features = features
        self.b_s = text.size(0)

    def generate_reinforcement_step(self, input_ids=None, max_len=1, repetition_penalty=1.2,
                                    start_token=50255, padoftext=50254):
        device = self.device
        T = self.T
        features = self.features
        if input_ids == None:
            caption = torch.LongTensor(self.b_s, 1).fill_(start_token).to(device)
        else:
            caption = input_ids[:, input_ids.size(-1) - 1:input_ids.size(-1)]
        lengths = torch.ones(self.b_s)
        results = torch.cat([input_ids, torch.LongTensor([[start_token]]).to(device)], dim=-1)
        with torch.no_grad():
            result = self.output(T, features, caption, lengths)

            result: torch.Tensor
            result = result[0]
            result = result.unsqueeze(1)
            i = result.size(-1)
            for b in range(results.size(0)):
                unique_samples, counts = torch.unique(results[b, :i], return_counts=True)
                for j in range(unique_samples.size(0)):
                    if unique_samples[j] != padoftext:
                        result[b][0][unique_samples[j]] /= repetition_penalty * counts[j]
            results[:, results.size(-1) - 2:results.size(-1)] = result.argmax(dim=2)

        return results, result

    def save_model(self, epoch, model_path="output"):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.encoder, f"{model_path}/en_{epoch}.pth")
        torch.save(self.decoder, f"{model_path}/de_{epoch}.pth")

    def load_model(self, epoch, model_path="output"):
        self.encoder = torch.load(f"{model_path}/en_{epoch}.pth", map_location=torch.device('cuda'))
        self.decoder = torch.load(f"{model_path}/de_{epoch}.pth", map_location=torch.device('cuda'))

    def template_search_predictions(self, image: torch.Tensor, text, feature_key_token: dict = None,
                                    feature_dict_token: dict = None, start_instruct_tokens=None):
        """
            Processes image and text data to generate predictions using an encoder-decoder architecture.
            Handles data parallelism across GPUs if available.

            Parameters:
            - image (torch.Tensor): Input image tensor.
            - text (torch.Tensor): Input text tensor.
            - feature_key_token (dict): Mapping of feature keys to tokens.
            - feature_dict_token (dict): Dictionary of feature tokens.
            - start_instruct_tokens (torch.Tensor): Tensor of start instruction tokens.

            Returns:
            - Tuple[torch.Tensor, dict]: A tuple containing the caption tensor and a dictionary of top results.
            """
        device = self.device
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
            T, features, (_, _) = encoder_parallel(image, text)
        else:
            T, features, (_, _) = self.encoder(image, text)
        text = copy.deepcopy(text)
        start_instruct_tokens = start_instruct_tokens.to(device)
        i = 0
        last_i = 0
        caption = None
        cache_key_top_results = {}
        cache_key_tensor_results = {}

        while True:

            if i >= text.size(1):
                break
            if text[0][i] not in start_instruct_tokens:
                i = i + 1
                continue
            if not caption == None:
                caption = torch.cat((caption, text[:, last_i:i]), dim=1)
            else:
                caption = text[:, 0:i]

            lengths = torch.ones(image.size(0)).fill_(caption.size(1))
            result = self.output(T, features, caption, lengths, image=image, text=text)

            result: torch.Tensor
            result = result[0]
            result = result.unsqueeze(0)

            sorted, indices = torch.sort(result[0][- 1], descending=True)

            i = i + 1

            current_key = None
            for k, v in feature_key_token.items():
                v_len = v.size(0)
                if v_len > 1:
                    if (v.to(device) == text[0][i:i + v_len]).sum() == v_len:
                        current_key = k
                        i = i + v_len - 1
                        break
                elif text[0][i] == v.item():
                    current_key = k
                    break
            if not current_key:
                raise Exception

            if current_key in cache_key_tensor_results.keys():
                pass
            else:

                cache_key_top_results[current_key] = {}
                cache_key_tensor_results[current_key] = {}
                for j in range(indices.size(0)):
                    flag = False
                    for kk, vv in feature_dict_token[current_key].items():
                        if indices[j] == vv[0].item():
                            vv = vv.to(device)
                            # cache_key_result[current_key] = kk
                            cache_key_top_results[current_key][kk] = j
                            cache_key_tensor_results[current_key][kk] = vv
                            flag = True
                    if flag:
                        break

                while len(cache_key_top_results[current_key].keys()) > 1:
                    tensor_key_index = 1
                    _, cache_tensor = list(cache_key_tensor_results[current_key].items())[0]
                    caption = torch.cat((caption, cache_tensor.unsqueeze(0)), dim=1)
                    lengths = torch.ones(image.size(0)).fill_(caption.size(1))
                    result = self.output(T, features, caption, lengths, image=image, text=text)

                    result: torch.Tensor
                    result = result[0]
                    result = result.unsqueeze(0)
                    _sorted, _indices = torch.sort(result[0][-1], descending=True)
                    tensor_key_index += 1
                    cache_tensor_result = {}
                    cache_top_result = {}
                    for j in range(_indices.size(0)):
                        flag = False
                        for kk, vv in cache_key_tensor_results[current_key].items():
                            if indices[j] == vv[
                                tensor_key_index].item():  # (v.to(device) == text[0][i:i + v_len]).sum() == v_len
                                cache_tensor_result[kk] = j
                                cache_top_result[kk] = (cache_key_top_results[current_key][kk] + j) * (
                                        tensor_key_index - 1) / tensor_key_index
                                flag = True
                        if flag:
                            break
                    cache_key_top_results[current_key] = cache_top_result
                    cache_key_tensor_results[current_key] = cache_tensor_result
            caption = torch.cat((caption, list(cache_key_tensor_results[current_key].items())[0][1].unsqueeze(0)),
                                dim=1)
            i = i + 1
            end_instruct_dict_tokens = {27422: 13, 5512: 11, 92: -1}
            for o_token, r_token in end_instruct_dict_tokens.items():
                if text[0][i] == o_token:
                    if r_token != -1:
                        caption = torch.cat((caption, torch.tensor([r_token]).to(device).unsqueeze(0)), dim=1)
                    break

            i += 1
            last_i = i
        return caption, cache_key_top_results

    def beam_search_predictions(self, image: torch.Tensor, text, max_len=128, eos_token=50256, beam_index=3, common=10):
        device = self.device
        text = text

        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
            T, features, (_, text_features) = encoder_parallel(image, text)
        else:
            T, features, (_, text_features) = self.encoder(image, text)
        caption = torch.LongTensor(image.size(0), 1).fill_(50255).to(device)
        states = None
        scores = self.output(T, features, caption, states=states)

        scores = scores[0]
        scores = scores.unsqueeze(1)
        scores_sort_values, scores_sort_indices = scores.sort(descending=True)
        scores_sort_indices = scores_sort_indices.squeeze(1)
        scores_sort_values = scores_sort_values.squeeze(1)

        current_length = 1
        next_args = [[None] * 3] * beam_index
        # scores_sort_values = scores_sort_values / current_length
        sampled_ids = [[] for _ in range(beam_index)]
        c = -1
        for i in range(beam_index):
            for c in range(c + 1, scores_sort_indices[0].size(0)):
                if scores_sort_indices[0][c].item() != eos_token:
                    break

            caption = scores_sort_indices[:, c:c + 1]
            score = scores_sort_values[:, c:c + 1]
            if states == None:
                next_args[i] = (caption, score, None)
            else:
                next_args[i] = (caption, score, tuple([s.clone() for s in states]))
            sampled_ids[i].append(caption)

        end_seqs = []

        while True:
            temp_args = [None] * (beam_index * beam_index)
            current_length += 1
            for i in range(len(next_args)):
                caption, score, states = next_args[i]
                scores = self.output(T, features, caption, states=states)
                scores = scores[0]
                scores = scores.unsqueeze(1)
                states: torch.Tensor
                scores_sort_values, scores_sort_indices = scores.sort(descending=True)
                scores_sort_indices = scores_sort_indices.squeeze(1)
                scores_sort_values = scores_sort_values.squeeze(1)
                scores_sort_values = scores_sort_values + score
                # scores_sort_values = scores_sort_values / current_length
                c = -1
                for j in range(beam_index):
                    for c in range(c + 1, scores_sort_indices[0].size()[0]):
                        if scores_sort_indices[0][c].item() == eos_token:
                            ca = torch.cat(sampled_ids[i], dim=1)
                            end_seqs.append((ca, scores_sort_values[0][c].item() * current_length / common))
                        else:
                            caption = scores_sort_indices[:, c:c + 1]
                            score = scores_sort_values[:, c:c + 1]
                            if states == None:
                                temp_args[i * beam_index + j] = (
                                    caption, score, None, score[0][0].item(), i)
                            else:
                                temp_args[i * beam_index + j] = (
                                    caption, score, tuple([s.clone() for s in states]), score[0][0].item(), i)
                            break
            temp_args = sorted(temp_args, reverse=True, key=lambda l: l[3])
            new_sampled_ids = [[] for i in range(beam_index)]
            for i in range(beam_index):
                next_args[i] = (temp_args[i][0], temp_args[i][1], temp_args[i][2])
                index = temp_args[i][4]
                caption = temp_args[i][0]
                new_captions = copy.deepcopy(sampled_ids[index])
                new_captions.append(caption)
                new_sampled_ids[i] = new_captions
            del sampled_ids
            sampled_ids = new_sampled_ids
            if current_length + 1 > max_len:
                for i in range(beam_index):
                    ca = torch.cat(sampled_ids[i], dim=1)
                    end_seqs.append((ca, next_args[i][1][0][0].item() * current_length / common))
                return sorted(end_seqs, reverse=True, key=lambda l: l[1])
