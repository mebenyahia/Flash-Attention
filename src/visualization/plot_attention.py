import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention(attn_weights, src_tokens, tgt_tokens):
    """
    Visualize the attention weights for a single head or aggregated heads.
    attn_weights: (tgt_seq_len, src_seq_len) or (n_heads, tgt_seq_len, src_seq_len)
    src_tokens: list of source tokens
    tgt_tokens: list of target tokens
    """
    if attn_weights.dim() == 3:
        # For multiple heads, we can average or pick one
        attn_weights = attn_weights.mean(dim=0)  # average over heads

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_weights.cpu().numpy(), xticklabels=src_tokens, yticklabels=tgt_tokens, cmap="Blues")
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.title("Attention Map")
    plt.show()
