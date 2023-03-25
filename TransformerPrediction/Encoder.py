#-- coding:UTF-8 --
import torch
import torch.nn as nn
from . import config as config
# import os,sys 
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

from .Attention import MultiHeadAttention, attn_pad_mask
from .FeedForwardNet import FeedForwardNet
from .positionalEncoding import PositionalEncoding



class EecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = FeedForwardNet()

    def forward(self, enc_input, enc_self_mask):
        """
        :param enc_input: [batch_size, src_len, d_model]
        :param enc_self_mask: [batch_size,seq_len,seq_len]
        :return:
        enc_outputs:[batch_size,seq_len,d_model]
        attn : [batch_len,n_heads,seq_len,seq_len]
        """
        # print(enc_input.shape)
        # print(enc_self_mask.shape)
        # print(enc_input, enc_self_mask)
        enc_outputs, attn = self.enc_self_attn(enc_input, enc_input, enc_input, enc_self_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


#
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.src_emb = nn.Embedding(config.src_vocab_size, config.d_model)
        self.pos_emb = PositionalEncoding(config.d_model)
        self.layers = nn.ModuleList([EecoderLayer() for _ in range(config.n_layers)])

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size , seq_len]
        :return:
        enc_outputs :[batch_size, seq_len, d_model]
        enc_self_attns : List [[batch_len,n_heads,seq_len,seq_len]]
        """
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size , seq_len, d_model]
        # zeros = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # enc_mask = []
        # for batch in enc_inputs:
        #     seq_comparison = []
        #     for seq in batch:
        #         # print(torch.equal(seq.float(), zeros))
        #         seq_comparison.append(torch.equal(seq.float(), zeros))
        #     enc_mask.append(seq_comparison)
        # enc_mask_tensor = torch.Tensor(enc_mask)
        # print("enc param shape:")
        # print(enc_inputs.shape)

        enc_outputs = enc_inputs
        # 位置编码
        enc_outputs = self.pos_emb(enc_outputs).to(config.device)  # [batch_size , seq_len, d_model]
        # print(enc_outputs)
        #  创建遮挡器mask
        enc_self_attn_mask = attn_pad_mask(enc_inputs, enc_inputs).to(config.device)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


#
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    # 预留一位填充0,模仿pad位置
    seq = torch.ones([batch_size, seq_len - 1])
    pad = torch.zeros([batch_size])
    pad = pad.unsqueeze(1)
    seq = torch.cat([seq, pad], dim=1).long()
    enc = Encoder()
    enc_outputs, enc_self_attns = enc(seq)

