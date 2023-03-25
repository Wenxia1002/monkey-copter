#-- coding:UTF-8 --
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config as config
from Transformer import Transformer
from dataSet import MKDataSet
import os

batch_index = 0
loss = 999
max_len = None
model = Transformer().to(config.device)
# 设置损失函数
kl_loss = nn.KLDivLoss(reduction="batchmean")
mse_loss = nn.MSELoss()
# 反向传播优化器
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
if os.path.exists(config.model_path):
    model.load_state_dict(torch.load(config.model_path))
    opt.load_state_dict(torch.load(config.opt_path))

dataset = MKDataSet(max_len=max_len)
loader = DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, shuffle=True, batch_size=config.batch_size)
for epoch in range(config.epoch):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        """
        # enc_inputs: [batch_size, src_len]
        # dec_inputs: [batch_size, tgt_len]
        # dec_outputs: [batch_size, tgt_len]
        """
        # TODO: 对enc_inputs，dec_inputs，dec_outputs进行处理

        enc_inputs = enc_inputs.to(config.device)
        dec_inputs = dec_inputs.to(config.device)
        dec_outputs = dec_outputs.to(config.device)
        outputs, enc_self_attn, dec_self_attn, dec_enc_attn = model(enc_inputs, dec_inputs)
        # [batch_size, tgt_seq_len, tgt_vocab_size] -> [batch_size * tgt_seq_len,tgt_vocab_size]
        # outputs = outputs.view(-1)
        # print(outputs)

        # [batch_size,tgt_seq_len] -> [batch_size * tgt_seq_len]
        # dec_outputs = dec_outputs.view(-1)
        # 计算loss
        # print(outputs.shape)
        # print(dec_outputs.shape)
        loss = mse_loss(outputs, dec_outputs)
        print(f"Epoch : {epoch + 1} Loss = {loss:.6f}")
        # 记录一下loss方便图形展示
        if batch_index % 10 == 0:
            with open(config.loss_path, 'a+') as f:
                f.write(f"{batch_index},{loss:.6f}\n")
        batch_index += 1
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 每次epoch之后持久化一下模型
    torch.save(model.state_dict(), config.model_path)
    torch.save(opt.state_dict(), config.opt_path)
