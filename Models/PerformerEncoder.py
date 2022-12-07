# -*- coding: utf-8 -*-
# @Time    : 2021/10/20 16:36
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: PerformerEncoder.py

import torch
import torch.nn as nn
from performer_pytorch import PerformerLM
from performer_pytorch import Performer

def get_attn_pad_mask(input_ids):
    pad_attn_mask_expand = torch.zeros_like(input_ids)
    batch_size, seq_len = input_ids.size()
    for i in range(batch_size):
        for j in range(seq_len):
            if input_ids[i][j] != 0:
                pad_attn_mask_expand[i][j] = 1

    return pad_attn_mask_expand.bool()
class PerformerEncoder(nn.Module):
    def __init__(self, config):
        super(PerformerEncoder, self).__init__()
        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        self.emb_dim = 64

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.performer_encoder = PerformerLM(
            num_tokens=self.emb_dim,
            dim=self.emb_dim,
            heads=8,
            depth=1,
            max_seq_len=config.max_len,
            reversible=True,
            local_attn_heads=4,  # 4 heads are local attention, 4 others are global performers
            local_window_size=config.max_len,  # window size of local attention
            # return_embeddings=True
                                            ).cuda()
        self.performer_encoder1 = Performer(
                                                        dim = 64,
                                                        depth = 1,
                                                        heads = 8,
                                                        causal = True,
                                                        dim_head = 8
                                                                        )

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
        # representation = self.performer_encoder(x)[:, 0, :].squeeze(1)
        # representation = self.performer_encoder1(x, mask=padding_mask)[:, 0, :].squeeze(1)
        representation = self.performer_encoder(x, mask=padding_mask)[:, 0, :].squeeze(1)

        output = self.classification(representation)

        return output, representation