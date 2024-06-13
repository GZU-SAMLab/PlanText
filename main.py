import argparse
import math
import os
import time
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

from data.data import LeafTemplateDataSet, collate_fn
from model.encoder2decoder import Encoder2Decoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from model.loss import MaskLoss
from model.optimizer import LRScheduler

from result.result import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(x, device=None, local_rank=None):
    if device:
        return x.to(device)
    elif local_rank:
        return x.to(device)
    elif torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def train(args):
    model = Encoder2Decoder(args.hidden_size).to(device)
    try:
        if args.debug:
            leaf = LeafTemplateDataSet(f"dataset/{args.annotation}/leaf_caption_debug.json", image_path=args.image_path,
                                       start_eq_end=False, not_template=args.not_template)

        else:
            leaf = LeafTemplateDataSet(f"dataset/{args.annotation}/leaf_caption_train.json",
                                       image_path=args.image_path,
                                       start_eq_end=False, not_template=args.not_template)
    except:
        print("data load error")
        raise
    data_loader = DataLoader(leaf, batch_size=args.batch_size, collate_fn=collate_fn)
    total_step = len(data_loader)
    vit_params = model.encoder.image_model.parameters()
    gpt_params = model.encoder.text_model.parameters()

    vit_optimizer = torch.optim.Adam(vit_params, lr=args.learning_rate_vit, betas=(args.alpha, args.beta))
    gpt_optimizer = torch.optim.Adam(gpt_params, lr=args.learning_rate_gpt, betas=(args.alpha, args.beta))

    params = list(model.encoder.image_fc.parameters()) + list(model.encoder.text_fc.parameters()) + list(
        model.encoder.alignment.parameters()) + list(model.encoder.projection.parameters()) + list(
        model.encoder.affine_a.parameters()) + list(model.encoder.affine_b.parameters()) + list(
        model.encoder.affine_c.parameters()) + list(model.decoder.parameters())

    if args.start_epoch > 1:
        model.load_model(args.start_epoch)
    start_time = time.time()
    optimizer = LRScheduler(
        filter(lambda x: x.requires_grad, params),
        args.hidden_size, 16000)
    LMcriterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        LMcriterion.cuda()
    for epoch in range(args.start_epoch, args.end_epoch + 1):

        for idx, (
                images, texts, targets, text2, target2, text_lengths, target_lengths, img_ids,
                template_ids) in enumerate(
            data_loader):

            images = to_device(images, device)
            texts = to_device(texts, device)
            targets = to_device(targets, device)

            model.train()
            model.zero_grad()

            lengths = [cap_len - 1 for cap_len in target_lengths]
            scores = model(images, texts, targets[:, :-1], lengths=lengths)
            targets = pack_padded_sequence(targets[:, 1:], lengths, batch_first=True)[0]
            loss = LMcriterion(scores[0], targets)

            loss.backward()

            optimizer.step()
            if epoch > args.pre_epoch:
                vit_optimizer.step()
                gpt_optimizer.step()

            if idx % args.log_iter == 0:
                log = 'Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.10f, Perplexity: %5.4f, UseTime: %f' % (
                    epoch, args.end_epoch, idx, total_step,
                    loss.item(),
                    np.exp(loss.item()),
                    time.time() - start_time)
                print(log)
                with open(f"log/train_{args.model_path}.log", "a", encoding="utf-8") as f:
                    f.write(f"{log}\n")
                    f.close()
        if epoch % args.save_epoch == 0:
            model.save_model(epoch, args.model_path)


#test log
@logger()
def test(args):
    model = Encoder2Decoder(args.hidden_size, device=device).to(device)

    try:
        if args.debug:
            leaf = LeafTemplateDataSet(f"dataset/{args.annotation}/leaf_caption_debug.json", image_path=args.image_path,
                                       start_eq_end=False, not_template=args.not_template)

        else:
            leaf = LeafTemplateDataSet(f"dataset/{args.annotation}/leaf_caption_test.json",
                                       image_path=args.image_path,
                                       start_eq_end=False, not_template=args.not_template)
    except:
        print("data load error")
        raise
    data_loader = DataLoader(leaf, batch_size=args.batch_size, collate_fn=collate_fn)

    model.load_model(args.test_epoch, args.model_path)
    model = model.to(device)
    model.eval()
    epoch_iterator = tqdm(data_loader, desc="Testing")
    for idx, (images, texts, targets, text2, target2, text_lengths, target_lengths, img_ids, template_ids) in enumerate(
            epoch_iterator):
        images = to_device(images, device)
        if args.not_template:

            text = torch.zeros(len(texts), max(text_lengths)).long().fill_(50254)
            texts = to_device(text, device)
        else:
            texts = to_device(texts, device)
        # captions = to_device(targets, device)
        if args.generate_repeated_penalties:
            end_seqs = model.generate_repeated_penalties(images, texts, device=device, max_len=72)
            decoded_preds = leaf.tokenizer.batch_decode(end_seqs, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
            args.log.write_cache(img_ids, template_ids, decoded_preds, "generate")
        if args.generate:
            end_seqs = model.generate(images, texts, max_len=72)
            decoded_preds = leaf.tokenizer.batch_decode(end_seqs, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
            args.log.write_cache(img_ids, template_ids, decoded_preds, "generate")
        if args.template_search:
            end_seqs, cache_key_top_results = model.template_search_predictions(images, texts,
                                                                                feature_key_token=leaf.feature_key_token,
                                                                                feature_dict_token=leaf.feature_dict_token,
                                                                                start_instruct_tokens=leaf.start_instruct_tokens)
            decoded_preds = leaf.tokenizer.batch_decode(end_seqs, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
            args.log.write_cache(img_ids, template_ids, decoded_preds, "template search", cache_key_top_results)
        if args.beam_search:
            end_seqs = model.beam_search_predictions(images, texts, beam_index=args.beam_index)

            decoded_preds = [
                leaf.tokenizer.batch_decode(end_seqs[i][0], skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False) for i in range(len(end_seqs))]

            args.log.write_cache(img_ids, template_ids, decoded_preds, "beam search")

    args.log.write()



def main(args):
    if args.test or args.generate or args.template_search or args.beam_search or args.generate_repeated_penalties:
        test(args)
    else:
        train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to manage various training and generation tasks.")

    # Debug and Testing options
    parser.add_argument('--debug', action="store_true", default=False, help='Enable debug mode')
    parser.add_argument('--test', action="store_true", default=False, help='Enable test mode')

    # Generation options
    parser.add_argument('--generate', action="store_true", default=False, help='Enable generation mode')
    parser.add_argument('--generate_repeated_penalties', action="store_true", default=False,
                        help='Enable generation with repeated penalties')
    parser.add_argument('--template_search', action="store_true", default=False, help='Enable template search mode')
    parser.add_argument('--beam_search', action="store_true", default=False, help='Enable beam search mode')
    parser.add_argument('--not_template', action="store_true", default=False, help='Disable text template usage')

    # Logging options
    parser.add_argument('--log_iter', type=int, default=10, help='Step size for printing log information')

    # Model fine-tuning options
    parser.add_argument('--pre_epoch', type=int, default=10,
                        help='Number of epochs to start fine-tuning GPT and VIT after')

    # Optimizer (Adam) parameters
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha parameter in Adam optimizer')
    parser.add_argument('--beta', type=float, default=0.999, help='Beta parameter in Adam optimizer')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate for the whole model')
    parser.add_argument('--learning_rate_gpt', type=float, default=1e-4, help='Learning rate for fine-tuning GPT')
    parser.add_argument('--learning_rate_vit', type=float, default=1e-4, help='Learning rate for fine-tuning VIT')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=512, help='Dimension of LSTM hidden states')

    # Training details
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch number')
    parser.add_argument('--end_epoch', type=int, default=10000, help='Ending epoch number')
    parser.add_argument('--save_epoch', type=int, default=1, help='Interval for saving the model (in epochs)')
    parser.add_argument('--test_epoch', type=int, default=20, help='Interval for testing the model (in epochs)')
    parser.add_argument('--beam_index', type=int, default=2, help='Beam index for beam search')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')

    # Paths and model names
    parser.add_argument('--model_path', type=str, default="output", help='Path to save the model')
    parser.add_argument('--annotation', type=str, default="annotation", help='Path to annotations')
    parser.add_argument('--image_path', type=str, default="images", help='Path to images')
    parser.add_argument('--model', type=str, default="PlanText", help='Name of the model to use')

    args = parser.parse_args()
    print(args)

    # Start training
    main(args)
