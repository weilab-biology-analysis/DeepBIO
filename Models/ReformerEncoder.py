# -*- coding: utf-8 -*-
# @Time    : 2021/10/20 11:01
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: ReformerEncoder.py

import torch
import torch.nn as nn
from reformer_pytorch import ReformerLM

def get_attn_pad_mask(input_ids):
    pad_attn_mask_expand = torch.zeros_like(input_ids)
    batch_size, seq_len = input_ids.size()
    for i in range(batch_size):
        for j in range(seq_len):
            if input_ids[i][j] != 0:
                pad_attn_mask_expand[i][j] = 1

    return pad_attn_mask_expand.bool()
class ReformerEncoder(nn.Module):
    def __init__(self, config):
        super(ReformerEncoder, self).__init__()
        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        self.emb_dim = 64

        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.reformer_encoder = ReformerLM(
                                                num_tokens = vocab_size,
                                                emb_dim = self.emb_dim,
                                                dim = 64,
                                                depth = 2,
                                                heads = 8,
                                                bucket_size=1,
                                                max_seq_len = config.max_len,
                                                fixed_position_emb = True,
                                                return_embeddings = True # return output of last attention layer
                                            ).cuda()

        self.classification = nn.Sequential(
                                    nn.Linear(64, 32),
                                    # nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    # nn.Dropout(0.5),
                                    nn.Linear(32, 2),
                                    )

    def forward(self, x):
        x = x.cuda()
        padding_mask = get_attn_pad_mask(x)
        # x = self.embedding(x)
        # representation = self.reformer_encoder(x)[:, 0, :].squeeze(1)
        representation = self.reformer_encoder(x, input_mask=padding_mask)[:, 0, :].squeeze(1)

        output = self.classification(representation)

        return output, representation

#Reformer使用实例
# DE_SEQ_LEN = 4096
# EN_SEQ_LEN = 4096
#
# encoder = ReformerLM(
#     num_tokens = 20000,
#     emb_dim = 128,
#     dim = 1024,
#     depth = 12,
#     heads = 8,
#     max_seq_len = DE_SEQ_LEN,
#     fixed_position_emb = True,
#     return_embeddings = True # return output of last attention layer
# ).cuda()
#
# decoder = ReformerLM(
#     num_tokens = 20000,
#     emb_dim = 128,
#     dim = 1024,
#     depth = 12,
#     heads = 8,
#     max_seq_len = EN_SEQ_LEN,
#     fixed_position_emb = True,
#     causal = True
# ).cuda()
#
# x  = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
# yi = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long().cuda()
#
# enc_keys = encoder(x)               # (1, 4096, 1024)
# yo = decoder(yi, keys = enc_keys)   # (1, 4096, 20000)