#-- coding:UTF-8 --
import torch
import torch.nn as nn
from . import config  as config
from .Attention import MultiHeadAttention, attn_pad_mask, attn_subsequence_mask
from .FeedForwardNet import FeedForwardNet
from .positionalEncoding import PositionalEncoding
from .Encoder import Encoder as Encoder

"""
Decoder部分要经过两次Attention阶段,第一次是对目标数据tgt进行Attention
"""


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        # 全连接层 + 残余项
        self.pos_ffn = FeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size,seq_tgt_len,d_model]
        :param enc_outputs: [batch_size,seq_src_len,d_model]
        :param dec_self_attn_mask: [batch_size,seq_tgt_len,seq_tgt_len]
        :param dec_enc_attn_mask: [batch_size,seq_tgt_len,seq_tgt_len]
        :return:
         dec_outputs: [batch_size, tgt_seq_len, d_model]
         dec_self_attn:[batch_len,n_heads,tgt_seq_len,tgt_seq_len]
         dec_enc_attn: [batch_len,n_heads,src_seq_len,tgt_seq_len]
        """
        # decoder对目标数据进行解析
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # 将encoder的输出和decoder本身解析的输出捏合到一起,进行decoder第二阶段的attention
        # Q : dec_outputs,K:enc_outputs,V: enc_outputs
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        #
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding 映射维度
        # self.tgt_emb = nn.Embedding(config.tgt_vocab_size, config.d_model)
        # 位置编码
        self.pos_emb = PositionalEncoding(config.d_model)
        # 多层Decoder
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config.n_layers)])

    #
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        :param dec_inputs: [batch_size,tgt_seq_len]
        :param enc_inputs: [batch_size,src_seq_len] # 仅需要他提供一个参数形状
        :param enc_outputs: [batch_size,src_seq_len,d_model]
        :return:
        dec_outputs:[batch_size, tgt_seq_len, d_model]
        """
        # zeros = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # enc_mask = []
        # for batch in enc_inputs:
        #     seq_comparison = []
        #     for seq in batch:
        #         # print(torch.equal(seq.float(), zeros))
        #         seq_comparison.append(torch.equal(seq.float(), zeros))
        #     enc_mask.append(seq_comparison)
        # enc_mask_tensor = torch.Tensor(enc_mask)
        # dec_mask = []
        # for batch in dec_inputs:
        #     seq_comparison = []
        #     for seq in batch:
        #         seq_comparison.append(torch.equal(seq.float(), zeros))
        #     dec_mask.append(seq_comparison)
        # dec_mask_tensor = torch.Tensor(dec_mask)

        # 维度放射
        # dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size,tgt_seq_len,d_model]
        # print("dec param shape:")
        # print(dec_inputs.shape, enc_inputs.shape, enc_outputs.shape)
        dec_outputs = dec_inputs
        # 添加位置编码
        dec_outputs = self.pos_emb(dec_outputs).to(config.device)

        # 针对填充值的mask True的位置为要遮挡的位置(python中True = 1,False=0)
        dec_self_attn_pad_mask = attn_pad_mask(dec_inputs, dec_inputs).to(config.device)
        # 针对控制输入顺序的mask [batch_size,tgt_seq_len,tgt_seq_len] ,需要遮挡的位置为1,不需要遮挡的位置为0
        dec_self_attn_subsequence_mask = attn_subsequence_mask(dec_inputs).to(config.device)
        # ****dec_self_mask****
        # pad的mask和控制输入顺序的mask合并到一起,返回值形状为[batch_size,tgt_seq_len,tgt_seq_len],
        # 其中值为0、1、2，
        # 0：不需要遮挡,
        # 1:dec_self_attn_pad_mask或者dec_self_attn_subsequence_mask中有一个触发遮挡一个未触发
        # 2:dec_self_attn_pad_mask和dec_self_attn_subsequence_mask全都触发遮挡
        dec_self_mask = dec_self_attn_pad_mask + dec_self_attn_subsequence_mask
        # 大于0的位置返回True(需要遮挡),不大于0的位置返回False(不需要遮挡)
        dec_self_attn_mask = torch.gt(dec_self_mask, 0).to(config.device)
        # encoder的第二阶段的attention,需要捏合encoder的输出和decoder的输入,所以需要如此传入
        dec_enc_attn_mask = attn_pad_mask(dec_inputs, enc_inputs)
        # 保存每次关联度,不参与后续运算,但是可以用来绘制热力图,方便观察
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


#
if __name__ == '__main__':
    batch_size, enc_seq_len, dec_seq_len, d_model = 2, 10, 12, config.d_model
    # 模拟encoder输入
    src_enc_seq = torch.tensor(range(0, 10)).unsqueeze(0).expand([batch_size, enc_seq_len])
    # 模拟encoder输出
    enc = Encoder()
    enc_outputs, enc_self_attns = enc(src_enc_seq)
    # 模拟一个decoder输入
    tgt_dec_seq = torch.tensor(range(10, 22)).unsqueeze(0).expand([batch_size, dec_seq_len])
    dec = Decoder()
    dec_outputs, dec_self_attns, dec_enc_attns = dec(tgt_dec_seq, src_enc_seq, enc_outputs)
    pass
