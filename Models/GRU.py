import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

def count_len(seqs, batch_len):
    lengths = []
    for seq in seqs:
        lengths.append(batch_len - np.bincount(seq.cpu())[0])
    max_len = seqs.shape[1]
    return max_len, lengths

class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()
        self.config = config
        self.embedding_dim = 128
        self.hidden_dim = 128

        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True, num_layers=2)
        self.classification = nn.Linear(self.hidden_dim, 2)


    def forward(self, x):
        x = x.cuda()
        max_len, lengths = count_len(x, self.config.max_len)

        x = self.embedding(x)
        # print(self.config.max_len)

        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        pack_representation, h_n = self.gru(packed_input, None)

        origin_representation, lens = pad_packed_sequence(pack_representation, batch_first=True)

        representation = torch.zeros(x.shape[0], self.hidden_dim).cuda()
        for index, seq_index in enumerate(lengths):
            representation[index] = origin_representation[index, seq_index-1, :]
        # print(h_n.shape)
        # print(representation.shape)
        # representation = h_n.squeeze(0)
        output = self.classification(representation)
        # print(output.shape)

        return output, representation
