import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class RNN_CNN(nn.Module):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self, config):

        super(RNN_CNN, self).__init__()
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = config.batch_size
        self.hidden_dim = 64
        self.embedding_dim = 128
        self.content_dim = 256

        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        # self.word_embeddings.weight = nn.Parameter(self.embeddings)
        # self.word_embeddings.weight.data.copy_(torch.from_numpy(self.embeddings))
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        # self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)
        self.hidden = self.init_hidden()

        self.conv = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.content_dim, kernel_size=4,
                              stride=self.embedding_dim)
        self.classification = nn.Linear(self.content_dim, 2)
        # self.properties.update({"content_dim": self.content_dim})

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x):
        # print(x.shape)
        # print(x.shape[0])

        x = x.cuda()
        embeds = self.word_embeddings(x)  # 64x200x300

        x = embeds.view(x.shape[1], x.shape[0], -1)
        # x = embeds.permute(1, 0, 2)  # 200x64x300

        hidden = self.init_hidden(x.shape[1])  # 1x64x128
        # print(hidden[0].shape)
        lstm_out, hidden = self.lstm(x,hidden)
        # input (seq_len, batch, input_size)
        # Outupts:output, (h_n, c_n) output:(seq_len, batch, hidden_size * num_directions)

        # lstm_out 200x64x128  lstm_out.permute(1,2,0):64x128x200
        representation = self.conv(lstm_out.permute(1, 2, 0))  ###64x256x1

        # y = self.conv(lstm_out.permute(1,2,0).contiguous().view(self.batch_size,128,-1))
        # y  = self.hidden2label(y.view(sentence.size()[0],-1))
        output = self.classification(representation.view(representation.size()[0], -1))  # 64x3
        return output, representation