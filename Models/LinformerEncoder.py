# -*- coding: utf-8 -*-
# @Time    : 2021/10/20 15:39
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: LinformerEncoder.py

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformerLM

def get_attn_pad_mask(input_ids):
    pad_attn_mask_expand = torch.zeros_like(input_ids)
    batch_size, seq_len = input_ids.size()
    for i in range(batch_size):
        for j in range(seq_len):
            if input_ids[i][j] != 0:
                pad_attn_mask_expand[i][j] = 1

    return pad_attn_mask_expand.bool()
class LinformerEncoder(nn.Module):
    def __init__(self, config):
        super(LinformerEncoder, self).__init__()
        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        self.emb_dim = 64

        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.linformer_encoder = LinearAttentionTransformerLM(
                                                                num_tokens = vocab_size,
                                                                dim = self.emb_dim,
                                                                heads = 8,
                                                                depth = 6,
                                                                max_seq_len =config.max_len,
                                                                reversible = True,
                                                                n_local_attn_heads = 4,
                                                                local_attn_window_size=config.max_len,
                                                                return_embeddings = True
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
        representation = self.linformer_encoder(x, input_mask=padding_mask)[:, 0, :].squeeze(1)

        output = self.classification(representation)

        return output, representation