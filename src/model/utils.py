import torch.nn as nn

def feed_forward(d_model, d_ff, dropout=0.1):
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout),
    )
