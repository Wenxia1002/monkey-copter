#-- coding:UTF-8 --
import torch
import torch.nn as nn
import numpy as np
from . import config

"""
本章主要实现 attention 以及 在encoder和decoder阶段的mask部分
"""


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q:[batch_size, n_heads, len_q, d_q]
        :param K:[batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v(=len_k), d_v]
        :param attn_mask:  [batch_size, n_heads, seq_q_len, seq_k_len]
        注意: 当encoder阶段和decoder第一个attention阶段的时候,Q=K=V(形状),当encoder的 第二个attention的时候,输入
        Q对应encoder的经过第一个个attention的输出,K和V对应的encoder端的输出
        :return:
        """
        # Q[batch_size, n_heads, len_q, d_k] x K.T[batch_size, n_heads, d_k, len_k] ->
        # scores [batch_size, n_heads,len_q,len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(config.d_k)
        # 将需要遮挡的位置(T OR F标记)用无穷小值覆盖,形状不变
        # scores.masked_fill_(attn_mask, -1e9)

        # 对最后一列做softmax 形状不变 : [batch_size, n_heads,len_q,len_k]
        attn = nn.Softmax(dim=-1)(scores)
        # [batch_size, n_heads,len_q,len_k] * [batch_size,n_heads,len_q, d_v] ->  [batch_size,n_heads,len_q, d_v]
        # TODO context = torch.matmul(attn.transpose(-1, -2), V) -> context = torch.matmul(attn, V)
        context = torch.matmul(attn, V)
        return context, attn

#
def attn_pad_mask(seq_q, seq_k):
    """
    用于decoder的attention和 encoder 第一个attention, K和Q的形状是一样的,
    但是当应用于encoder的第二个attention的时候,由于
    Q是encoder提供形状为[batch_size,src_seq_len,n_model],而
    K是通过encoder的第一个attention产生形状为[batch_size,tgt_seq_len,n_model],
    所以Q*K的转置之后形状变成[batch_size,src_seq_len,tgt_seq_len],故而mask也需要做相同形状
    形状
    :param seq_q: [batch_size,seq_q_len]
    :param seq_k: [batch_size,seq_k_len]
    :return:
    返回一个大小为
    [batch_size, seq_q_len, seq_k_len]
    组成元素为TRUE或者FALSE的矩阵,TRUE代表匹配到0的位置
    """
    # print( seq_q.size(), seq_k.size())
    batch_size, len_q, temp = seq_q.size()
    # print(seq_q.shape)
    # batch_size, len_q = seq_q.size()
    # batch_size, len_k = seq_k.size()
    # <PAD>的index为0，匹配到数据中包含0的位置，给与True或者False

    zeros = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(config.device)
    enc_mask = []
    for batch in seq_q:
        seq_comparison = []
        for seq in batch:
            # print(torch.equal(seq.float(), zeros))
            # print(seq.float())
            atom_comparison = torch.equal(seq, zeros)
            # print(atom_comparison)
            seq_comparison.append(atom_comparison)
        enc_mask.append(seq_comparison)
    pad_attn_mask = torch.Tensor(enc_mask).unsqueeze(1)

    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # pad_attn_mask = seq_k.data.eq([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unsqueeze(1)
    # print(pad_attn_mask)
    # 用存在元素填充至[batch_size, seq_len, seq_len]大小
    # print(pad_attn_mask)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_q)
    return pad_attn_mask


def attn_subsequence_mask(in_seq):
    """
    本mask只在encoder阶段使用，encoder的输入是一次性输入到attention中的，
    比如 [<BOS>,鸡,你,太,美,<EOS>] 但实际上我们想要的是
    输入<BOS> -> 输出 鸡
    输入 鸡 -> 输出 你
    输入 你 -> 输出 太
    输入 太 -> 输出 美
    输入 美 -> 输出 <EOS>
    所以输入矩阵要养着对角线进行遮挡,本方法就是产生一个这样的三角矩阵
    :param in_seq: [batch_size,tgt_len]
    :return:
    """

    attn_shape = [in_seq.size(0), in_seq.size(1), in_seq.size(1)]
    # 产生三角矩阵,k=1对角线向上移动1个单位
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


# 
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(config.d_model, config.d_q * config.n_heads, bias=False)
        self.W_K = nn.Linear(config.d_model, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_model, config.d_v * config.n_heads, bias=False)
        self.fc = nn.Linear(config.d_v * config.n_heads, config.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # [batch_size, len_q, d_model] -> [batch_size, len_q , d_q * n_heads]
        # print(input_Q.float())
        Q = self.W_Q(input_Q.float())
        K = self.W_K(input_K.float())
        V = self.W_V(input_V.float())
        # view 哪行为-1即为自适应  [batch_size, len_q , d_q * n_heads] ->  [batch_size, len_q ,n_heads ,d_q]
        Q = Q.view(batch_size, -1, config.n_heads, config.d_q)
        K = K.view(batch_size, -1, config.n_heads, config.d_k)
        V = V.view(batch_size, -1, config.n_heads, config.d_v)
        # [batch_size, len_q ,n_heads ,d_q] -> [batch_size, n_heads ,len_q ,d_q]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 在1位置扩充一个维度 :  [batch_size, seq_len, seq_len] -> [batch_size,1, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1)
        # 将 [seq_len, seq_len] 部分复制 n_heads份 [batch_size,1, seq_len, seq_len] -> [batch_size,n_head, seq_len, seq_len]
        attn_mask = attn_mask.repeat(1, config.n_heads, 1, 1)
        # Q:[batch_size, n_heads, len_q, d_k] attn_mask:[batch_size, n_heads, seq_len, seq_len] ->
        # context : [batch_len,n_heads,seq_len,d_v] ,attn:[batch_len,n_heads,seq_len,seq_len] 该返回值仅用于热力图展示,观察词根之间的关联程度

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        #  [batch_len,n_heads,seq_len,d_v] -> [batch_len,seq_len,n_heads,d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, config.n_heads * config.d_v)
        #  [batch_size, len_q, n_heads * d_v] -> [batch_size, len_q, d_model]
        output = self.fc(context)
        # [batch_len,seq_len,d_model]
        return nn.LayerNorm(config.d_model).to(config.device)(output + residual.float()), attn


#
if __name__ == '__main__':
    # 测试:创建一个长度为2,seq长度为10的,8个多头注意力机制的数据集
    batch_size, n_heads, seq_len = 2, config.n_heads, 10
    # 1.先创建[batch_size,seq_len - 1] 全为1的矩阵
    Q = torch.ones([batch_size, seq_len - 1])
    # 2.创建[batch_size] 的全0向量,模拟<PAD>位置数据
    p = torch.zeros([batch_size])
    p = p.unsqueeze(-1)
    # 将1,2组合起来,[batch_size,seq_len] 组成每行最后位置为0的矩阵
    in_attn_Q = torch.cat((Q, p), 1).long()  # 模仿decoder输入
    in_attn_K = in_attn_V = torch.cat((in_attn_Q, p), 1).long()  # 模仿encoder输入
    # 对KQV进行Embedding操作,500随便给的,d_model代表每个位置被映射为多少维度
    Q = nn.Embedding(500, config.d_model)(in_attn_Q)
    K = V = nn.Embedding(500, config.d_model)(in_attn_K)
    # 创建 mask
    mask = attn_pad_mask(in_attn_Q, in_attn_K)
    # 计算attn
    attn = MultiHeadAttention()
    context, attn = attn(Q, K, V, mask)
    pass

"""
	token				Embeding + Positional Encoding (长度为n_model)

	Chicken		[-2.5183,  0.7094, -1.0897,  0.0957,  0.3010,  2.0954,  0.4646,-1.1324,  1.7186,  1.1405]
	,		[-1.3225,  1.0005,  0.7154, -1.5705, -1.3154, -1.8551, -1.7161, 0.1276,  1.9049, -1.5616]
	you		[-0.0383, -1.4279, -0.0538,  0.1469, -1.0270, -0.5083,  0.1013, 1.7076, -0.1840, -0.3653]
	are		[-0.9482, -1.2001, -0.4321, -0.6330, -1.8740, -0.1657,  2.0102, -1.2926, -0.0996, -0.0832]
	so		[-0.8802, -0.8078, -1.2101,  0.0869, -0.3669, -0.2666, -0.0729, 0.4369, -0.1241,  2.5872]
	beautiful	[-0.4022, -0.0721,  0.7918, -0.1862,  0.7195,  0.6027, -0.0784, 0.2827, -1.0347, -0.9485]
	!		[ 0.1809,  1.8887, -0.8190,  1.4965, -1.4000, -0.2994,  0.1361, -2.5050,  0.5601, -0.3250]
	<PAD>		[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000,  0.0000

"""
