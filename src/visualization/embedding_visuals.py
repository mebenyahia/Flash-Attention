import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

def plot_embeddings(emb_matrix, title="Embeddings"):
    """
    emb_matrix: (vocab_size, d_model)
    This is a direct visualization. For deeper analysis, apply PCA/TSNE first.
    """
    vocab_size, d_model = emb_matrix.shape
    sample_size = min(100, vocab_size)
    subset = emb_matrix[:sample_size, :]

    plt.figure(figsize=(10, 6))
    sns.heatmap(subset, cmap="viridis")
    plt.title(title)
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Vocab Indices (sampled)")
    plt.show()
