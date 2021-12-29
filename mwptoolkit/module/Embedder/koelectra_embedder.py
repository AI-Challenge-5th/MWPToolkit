# -*- encoding: utf-8 -*-
# @Author: JW
# @Time: 2021:10:15
# @File: koelectra_embedder.py


import torch
from torch import nn
from transformers import ElectraModel

class KoElectraEmbedder(nn.Module):
    def __init__(self,input_size,pretrained_model_path):
        super(KoElectraEmbedder,self).__init__()
        self.koelectra=ElectraModel.from_pretrained(pretrained_model_path)

    
    def forward(self,input_seq,attn_mask):
        output=self.koelectra(input_seq,attention_mask = attn_mask)[0]
        return output
    
    def token_resize(self,input_size):
        self.koelectra.resize_token_embeddings(input_size)
