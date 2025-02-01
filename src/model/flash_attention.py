"""
Implements FlashAttention using the flash-attn library for GPU-optimized attention.
Requires input: [B, S, n_heads, head_dim], in fp16 or bf16 if running on GPU.
"""

import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func

class FlashAttention(nn.Module):
    def __init__(self, dropout: float = 0.0, causal: bool = False):
        """
        Args:
            dropout (float): Dropout probability for attention weights.
            causal (bool): If True, apply causal masking (for decoder).
        """
        super().__init__()
        self.dropout = dropout
        self.causal = causal

    def forward(self, query, key, value, attention_mask=None):
        """
        query, key, value: (batch_size, seq_len, n_heads, head_dim)
        Returns:
            output: (batch_size, seq_len, n_heads, head_dim)

        flash_attn_func now requires 4D tensors in half precision/bfloat16 on GPU.
        """
        # We do not flatten n_heads. The library references q.size(3) for head_dim.

        out = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,
            causal=self.causal
        )
        return out
