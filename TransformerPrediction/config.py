#-- coding:UTF-8 --
import torch
import pickle

"""
配置参数
"""
# with open("./data/s03_8000_index_and_word_dict.plk", "rb") as f:
#     src_index_2_word, \
#     src_word_2_index, \
#     tgt_index_2_word, \
#     tgt_word_2_index = pickle.load(f)
# # 英文字典长度
# src_vocab_size = len(src_index_2_word)
# # 汉语字典长度
# tgt_vocab_size = len(tgt_index_2_word)

len_output = 11

drop = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open("./data/s03_8000_index_and_word_dict.plk", "rb") as f:
#     index_2_word_en, word_2_index_en, index_2_word_ch, word_2_index_ch = pickle.load(f)
"""
d_q = d_k = 64 
d_v = 64
# input 乘以w_q和w_k时候扩展的维度
例如,input_X[batch_size,seq_len] * W_Q[seq_len,d_q]-> Q[batch_size,d_q]
KV同理
"""
d_q = d_k = 64
d_v = 64
d_model = 6  # 每个token转换成多少维度,等同于Embedding
n_heads = 8  # 多头机制

d_ff = 2048  # 全连接层发散的维度
# encoder和decoder的层数
n_layers = 6

model_path = "./model/transformer_model2.pkl"
opt_path = "./model/transformer_optimizer2.pkl"

loss_path = "./model/loss2.txt"
# 总的训练次数,每次跑完所有训练数据算一次训练次数
epoch = 100
# 每次从 DataLoader 里拉取多少数据
batch_size = 100
