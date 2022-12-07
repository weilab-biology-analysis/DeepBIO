import torch
import torch.nn as nn
import torch.nn.functional as F

# 目的是构造出一个注意力判断矩阵，一个[batch_size, seq_len, seq_len]的张量
# 其中参与注意力计算的位置被标记为FALSE，将token为[PAD]的位置掩模标记为TRUE
def get_attn_pad_mask(input_ids):
    pad_attn_mask_expand = torch.zeros_like(input_ids)
    batch_size, seq_len = input_ids.size()
    for i in range(batch_size):
        for j in range(seq_len):
            if input_ids[i][j] != 0:
                pad_attn_mask_expand[i][j] = 1

    return pad_attn_mask_expand

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        self.emb_dim = 128

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)

        self.fnn = nn.Linear(self.emb_dim * config.max_len, self.emb_dim)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.emb_dim, 2)
        )

    def forward(self, x):
        x = x.cuda()

        # padding_mask = get_attn_pad_mask(x).permute(1,0)
        # print(x.shape)
        x = self.embedding(x)
        # src_key_padding_mask = padding_mask
        # representation = self.transformer_encoder(x,)[:, 0,:].squeeze(1)
        representation = self.transformer_encoder(x,)
        representation = representation.view(representation.shape[0], -1)
        representation = self.fnn(representation)
        # print(representation.shape)
        # print(representation)
        output = self.classifier(representation)

        return output, representation

