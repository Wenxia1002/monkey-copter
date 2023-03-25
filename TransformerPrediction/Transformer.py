#-- coding:UTF-8 --
import torch
import torch.nn as nn 
from . import config as config
# import config as config
from . import Encoder as encoder
from . import Decoder as decoder
# from Encoder import Encoder
# from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder.Encoder().to(config.device)
        self.decoder = decoder.Decoder().to(config.device)
        # 将最后的结果映射为中文词根字典的长度
        self.projection = nn.Linear(config.d_model, config.d_model, bias=False).to(config.device)

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: [batch_size,src_seq_len]
        :param dec_inputs: [batch_size,tgt_seq_len]
        :return:
        """
        # enc_outputs: [batch_size,src_seq_len,d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_output : [batch_size, tgt_seq_len, d_model]
        dec_output, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_output)  # [batch_size,tgt_seq_len,tgt_vocab_size]
        # TODO
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

    def translate(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # 创建decoder input
        # dec_inputs = torch.tensor([[[0.0,0.0,0.0,0.0,0.0,0.0]] + enc_inputs[1:]])
        # print(dec_inputs)
        dec_output, dec_self_attns, dec_enc_attns = self.decoder(enc_inputs, enc_inputs, enc_outputs)
        # print(dec_output)
        # dec_inputs = torch.tensor([[[0.0,0.0,0.0,0.0,0.0,0.0]]])
        # for i in range(config.len_output):
        #     dec_output, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        #     dec_logits = self.projection(dec_output)
        #     # print(dec_logits)
        #     # values, indices = torch.max(dec_logits, dim=2)
        #     # indices = indices[0][-1].unsqueeze(0).unsqueeze(0)
        #     # [1, tgt_seq_len]
        #     dec_inputs = torch.cat([dec_inputs, dec_logits], dim=1)
        #     # if indices == config.tgt_word_2_index["<EOS>"]:
        #     #     break
        return dec_output[0][-1]


if __name__ == '__main__':
    # batch_size, src_size, tgt_size = 2, 10, 12
    # encoder_inputs = torch.ones([batch_size, src_size]).long()
    # decoder_inputs = torch.ones([batch_size, tgt_size]).long()
    model = Transformer()
    # decoder_logits, _, _, _ = model(encoder_inputs, decoder_inputs)
    # 翻译测试
    #encoder_inputs = torch.ones([1, src_size]).long()
    #dec_inputs = model.translate(encoder_inputs)

    pass

#
