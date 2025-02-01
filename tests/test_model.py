import pytest
import torch
from src.model.transformer import Transformer

def test_transformer_forward():
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        max_len=16,
        dropout=0.1
    )
    src = torch.randint(0, 100, (2, 10))  # (batch=2, seq_len=10)
    tgt = torch.randint(0, 100, (2, 10))  # (batch=2, seq_len=10)
    out = model(src, tgt)
    assert out.shape == (2, 10, 100), "Output shape mismatch"
