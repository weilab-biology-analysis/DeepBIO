import torch.nn as nn
import torch.nn.functional as F
import torch

class TextRCNN(nn.Module):

    def __init__(self, config):
        super(TextRCNN, self).__init__()

        self.embedding_dim = 128
        self.hidden_dim = 128
        self.batch_size = config.batch_size
        self.num_layers = 2
        self.use_gpu = torch.cuda.is_available()

        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=0.3)
        self.W2 = nn.Linear(2 * self.hidden_dim + self.embedding_dim, self.hidden_dim * 2)
        self.classification = nn.Linear(self.hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):

        text = x.cuda()
        # text: [seq_len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded: [seq_len, batch size, emb dim]

        outputs, _ = self.rnn(embedded)
        # outputs: [seq_len， batch_size, hidden_dim * bidirectional]

        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, hidden_dim * bidirectional]
        # print(outputs.shape)

        embedded = embedded.permute(1, 0, 2)
        # embeded: [batch_size, seq_len, embeding_dim]

        x = torch.cat((outputs, embedded), 2)
        # x: [batch_size, seq_len, embdding_dim + hidden_dim * bidirectional]

        y2 = torch.tanh(self.W2(x)).permute(1, 2, 0)
        # y2: [batch_size, hidden_dim * bidirectional, seq_len]

        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
        # y3: [batch_size, hidden_dim * bidirectional]
        # print(y3.shape)

        output = self.classification(y3)
        # print(output.shape)

        return output, y3