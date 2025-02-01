import torch
import torch.nn as nn
from .flash_attention import FlashAttention
from .utils import feed_forward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.self_attn = FlashAttention(dropout=dropout, causal=True)
        self.cross_attn = FlashAttention(dropout=dropout, causal=False)

        self.self_attn_proj = nn.Linear(d_model, d_model)
        self.cross_attn_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff = feed_forward(d_model, d_ff, dropout)

        # Q/K/V projection for self-attn
        self.qkv_proj_self = nn.Linear(d_model, 3 * d_model)
        # Q, KV projection for cross-attn
        self.qkv_proj_cross_q = nn.Linear(d_model, d_model)
        self.qkv_proj_cross_kv = nn.Linear(d_model, 2 * d_model)

    def forward(self, x, enc_out):
        """
        x: (batch_size, tgt_seq_len, d_model)
        enc_out: (batch_size, src_seq_len, d_model)
        """
        bsz, tgt_len, _ = x.shape
        _, src_len, _ = enc_out.shape
        head_dim = self.d_model // self.n_heads

        # 1) Masked Self-Attn
        qkv_self = self.qkv_proj_self(x)
        qkv_self = qkv_self.view(bsz, tgt_len, 3, self.n_heads, head_dim)
        q_self = qkv_self[:, :, 0]
        k_self = qkv_self[:, :, 1]
        v_self = qkv_self[:, :, 2]

        residual = x
        attn_out = self.self_attn(q_self, k_self, v_self)
        attn_out = attn_out.reshape(bsz, tgt_len, self.d_model)
        attn_out = self.self_attn_proj(attn_out)
        x = self.norm1(residual + attn_out)

        # 2) Cross-Attn
        residual = x
        q_cross = self.qkv_proj_cross_q(x)
        q_cross = q_cross.view(bsz, tgt_len, self.n_heads, head_dim)

        kv_cross = self.qkv_proj_cross_kv(enc_out)
        kv_cross = kv_cross.view(bsz, src_len, 2, self.n_heads, head_dim)
        k_cross = kv_cross[:, :, 0]
        v_cross = kv_cross[:, :, 1]

        attn_out_cross = self.cross_attn(q_cross, k_cross, v_cross)
        attn_out_cross = attn_out_cross.reshape(bsz, tgt_len, self.d_model)
        attn_out_cross = self.cross_attn_proj(attn_out_cross)
        x = self.norm2(residual + attn_out_cross)

        # 3) Feed Forward
        residual = x
        x = self.ff(x)
        x = self.norm3(residual + x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x
