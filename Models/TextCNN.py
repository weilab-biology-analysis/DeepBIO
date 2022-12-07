import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        self.filter_sizes = [1,2,4,8]
        self.embedding_dim = 128
        dim_cnn_out = 128
        filter_num = 128

        if config.type == "DNA" or config.type == "RNA":
            vocab_size = 6
        elif config.type == "prot":
            vocab_size = 26

        # self.filter_sizes = [int(fsz) for fsz in self.filter_sizes.split(',')]
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, self.embedding_dim)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.2)

        self.linear = nn.Linear(len(self.filter_sizes) * filter_num, dim_cnn_out)
        self.classification = nn.Linear(dim_cnn_out, 2)  # label_num: 28


    def forward(self, x):
        x = x.cuda()
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        # print('raw x', x.size())
        input_ids = x
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        # print('embedding x', x.size())

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)
        # print('view x', x.size())

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]
        # print(x)
        # print('conv x', len(x), [x_item.size() for x_item in x])

        # 平均的方法进行attention值的获取
        # compute = [x_item.detach() for x_item in x][:2]
        # count = torch.zeros_like(input_ids)
        # attention = torch.zeros_like(input_ids)
        # for index, length in enumerate(self.filter_sizes[:2]):
        #     filter_x = compute[index]  # 第i个卷积核的结果
        #     # print(filter_x.shape) torch.Size([32, 128, 209, 1])
        #     filter_x_squeeze = torch.squeeze(filter_x)  # 将最后一维压缩 (batch_size, 128, 209)
        #     # print(filter_x_squeeze.shape) torch.Size([32, 128, 209])
        #     sum_filter_x = torch.sum(filter_x_squeeze, dim=1)  # (batch_size, w)
        #     # print(sum_filter_x)
        #     # print(sum_filter_x.shape) torch.Size([32, 209])
        #     for i in range(sum_filter_x.size(1)):
        #         for sum_index in range(length):
        #             count[:, i + sum_index] = count[:, i + sum_index] + 1
        #             # print(attention[:,sum_index].shape) torch.Size([32])
        #             # print(sum_filter_x[:,i+sum_index].shape) torch.Size([32])
        #             attention[:, i + sum_index] = attention[:, i + sum_index] + sum_filter_x[:, i]
        # # print(count)
        # # print(attention)
        # attention = attention / count
        # print(attention)

        # 取max得到attention值的分数
        # compute = [x_item.detach() for x_item in x]
        # attention = torch.zeros_like(input_ids, dtype=torch.float)
        #
        # acid_count = torch.zeros(input_ids.size(0), 28)
        #
        # for index, length in enumerate(self.filter_sizes[:2]):
        #     filter_x = compute[index]  # 第i个卷积核的结果
        #     # print(filter_x.shape) torch.Size([32, 128, 209, 1])
        #     filter_x_squeeze = torch.squeeze(filter_x)  # 将最后一维压缩 (batch_size, 128, 209)
        #     # print(filter_x_squeeze.shape) torch.Size([32, 128, 209])
        #     max_filter_x_squeeze = torch.max(filter_x_squeeze, dim=2)
        #     # print(max_filter_x_squeeze)
        #     # print(max_filter_x_squeeze[1].shape)
        #     for batch in range(max_filter_x_squeeze[1].size(0)):
        #         for index, max_i in enumerate(max_filter_x_squeeze[1][batch]):
        #             for sum_index in range(length):
        #                 # print(attention[:, sum_index].shape)
        #                 # print(max_filter_x_squeeze[0][:, index].shape)
        #                 # print(attention[:, sum_index][1].dtype)
        #                 # print(max_filter_x_squeeze[0][:, index][1].dtype)
        #                 # print(max_filter_x_squeeze[0][batch][index])
        #                 acid_count[batch][input_ids[batch][max_i + sum_index]] = acid_count[batch][input_ids[batch][max_i + sum_index]] + 1
        #                 attention[batch, max_i + sum_index] = attention[batch, max_i + sum_index] + max_filter_x_squeeze[0][batch][index]
        # # print(acid_count)
        # # print(attention)

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # print('max_pool2d x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # print('flatten x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)
        # print('concat x', x.size()) torch.Size([320, 1024])

        # dropout层
        x = self.dropout(x)

        # 全连接层
        representation = self.linear(x)
        output = self.classification(representation)

        return output, representation
