# -*- coding: utf-8 -*-
# @Time    : 2021/10/18 19:50
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: DNN.py

import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.config = config
        self.embedding_dim = 128

        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        # self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        # self.fc = nn.Sequential(nn.Linear(self.embedding_dim, 64),
        #                                     nn.Dropout(0.2),
        #                                     nn.ReLU(),
        #                                     nn.Linear(64, 32))
        self.classification = nn.Linear(self.config.max_len, 2)

    def forward(self, x):
        x = x.cuda().float()

        # x = self.embedding(x)
        # x = x[:, 0, :].squeeze(1)
        # representation = self.fc(x)

        output = self.classification(x)
        # print(output.shape)

        return output, x