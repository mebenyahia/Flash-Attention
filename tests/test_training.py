import pytest
import torch
from src.training.regularization import label_smoothing_loss

def test_label_smoothing_loss():
    logits = torch.randn(2, 3, 10)  # (batch=2, seq_len=3, vocab_size=10)
    target = torch.randint(0, 10, (2, 3))
    loss = label_smoothing_loss(logits, target, smoothing=0.1)
    assert loss >= 0.0, "Loss should be non-negative"
