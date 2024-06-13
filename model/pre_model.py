from transformers import ViTModel, GPT2Model, GPT2Tokenizer
import json
vit = ViTModel.from_pretrained('./model/pre_gpt/', output_attentions=True)
gpt2 = GPT2Model.from_pretrained("./model/pre_gpt/")

tokenizer=GPT2Tokenizer.from_pretrained('./model/pre_gpt/')
tokenizer.pad_token = "<|padoftext|>"
tokenizer.bos_token = "<|startoftext|>"