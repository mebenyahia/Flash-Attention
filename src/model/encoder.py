import torch
import torch.nn as nn
from .flash_attention import FlashAttention
from .utils import feed_forward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.self_attn = FlashAttention(dropout=dropout, causal=False)
        self.attn_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = feed_forward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Projection for Q/K/V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        We'll shape Q, K, V => [B, S, n_heads, head_dim].
        """
        bsz, seq_len, _ = x.shape
        head_dim = self.d_model // self.n_heads

        # 1) Project Q, K, V
        qkv = self.qkv_proj(x)  # (bsz, seq_len, 3*d_model)
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # 2) Self-attention
        residual = x
        attn_out = self.self_attn(q, k, v)  # (bsz, seq_len, n_heads, head_dim)
        attn_out = attn_out.reshape(bsz, seq_len, self.d_model)
        attn_out = self.attn_proj(attn_out)
        x = self.norm1(residual + attn_out)

        # 3) Feed Forward
        residual = x
        x = self.ff(x)
        x = self.norm2(residual + x)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
