# coding: utf-8
from torch import nn
from torch.nn import functional as F
import torch


class CNNReg(nn.Module):
    def __init__(self, vocaNum, embedding_dim):
        super(CNNReg, self).__init__()
        self.kernel_size = [2, 3, 4, 5]
        self.channel_out = 10
        self.embedding = nn.Embedding(vocaNum, embedding_dim)
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(1, self.channel_out, (k, embedding_dim)) for k in self.kernel_size])
        self.linear1 = nn.Linear(self.channel_out*len(self.kernel_size), 10)
        self.linear2 = nn.Linear(10, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        embed = self.embedding(x)   # (N, W, D)
        embed = embed.unsqueeze(1)  # (N,1,W,D), 1: channel_in

        # [(N,Channel_out,W,1), ...] * len(Kernel_size)
        feature_maps = [F.relu(conv(embed)) for conv in self.conv1]
        
        # [(N,Channel_out,W), ...] * len(Kernel_size)
        feature_maps = [feature_map.squeeze(3) for feature_map in feature_maps]

        # [(N, Channel_out), ...] * len(Kernel_size)
        pooled_output = [F.max_pool1d(feature_map, feature_map.size(2)) for feature_map in feature_maps]
        output = torch.cat(pooled_output, 1)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)
        return output


def makeBatch(batch_sequences, max_len=100):
    batch_sequences = [torch.Tensor(sequence) for sequence in batch_sequences]
    lengths = [len(sequence) for sequence in batch_sequences]
    output = torch.zeros(len(batch_sequences), max_len)
    for i, sequence in enumerate(batch_sequences):
        length = lengths[i]
        output[i, :length] = sequence[:length]
    output = output.long()
    return output
