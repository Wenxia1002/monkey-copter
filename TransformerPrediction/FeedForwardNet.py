#-- coding:UTF-8 --
import torch
import torch.nn as nn
from . import config as config

"""
本篇对应attention中的FeedForward部分,encoder一次,decoder一次
"""
#

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model,bias=False)
        )

    def forward(self, inputs):
        """
        :param inputs: [batch_size,seq_len,d_model]
        :return:outputs : [batch_size, seq_len, d_model]
        """
        residual = inputs
        outputs = self.fc(inputs)
        return nn.LayerNorm(config.d_model).to(config.device)(outputs + residual)


#
if __name__ == '__main__':
    batch_size, seq_len, d_model = 2, 10, 512
    inputs = torch.ones([batch_size, seq_len, d_model])
    ffd = FeedForwardNet()
    out = ffd(inputs)
    pass
    #
