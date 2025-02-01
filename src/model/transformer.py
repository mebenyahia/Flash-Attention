import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .embeddings import TokenEmbedding, PositionalEncoding

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        max_len,
        dropout=0.1
    ):
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_encoder_layers,
            n_heads,
            d_ff,
            max_len,
            dropout
        )

        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len)
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_decoder_layers,
            n_heads,
            d_ff,
            max_len,
            dropout
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # src, tgt: (batch_size, seq_len)
        enc_inp = self.src_embedding(src)
        enc_inp = self.src_pos_encoding(enc_inp)
        enc_out = self.encoder(enc_inp)

        dec_inp = self.tgt_embedding(tgt)
        dec_inp = self.tgt_pos_encoding(dec_inp)
        dec_out = self.decoder(dec_inp, enc_out)

        logits = self.output_proj(dec_out)  # (batch_size, tgt_seq_len, vocab_size)
        return logits
