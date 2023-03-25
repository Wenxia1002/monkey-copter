#-- coding:UTF-8 --
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns

"""

"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model 应设置为偶数!"

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        a = position * div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        :return [batch_size,seq_len,d_model]
        '''
        x = x.transpose(1, 0)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).transpose(1, 0)


#
if __name__ == '__main__':
    batch_size = 2
    d_model = 100  # 必须确保为偶数
    max_len = 100
    x = torch.zeros([batch_size, max_len, d_model], dtype=torch.float)
    pe = PositionalEncoding(d_model, max_len=max_len)
    pe_coder = pe(x)
    pe_coder1 = pe_coder[0, :, :].squeeze(1)
    pe_coder1 = pe_coder1.numpy()
    plt.figure(figsize=(10, 10))
    sns.heatmap(pe_coder1)
    plt.title("Sinusoidal Function")
    plt.xlabel("d_model")
    plt.ylabel("POS")
    plt.show()

#
