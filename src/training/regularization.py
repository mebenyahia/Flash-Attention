import torch
import torch.nn.functional as F

def label_smoothing_loss(logits, target, smoothing=0.1):
    """
    Computes label-smoothing cross-entropy loss.
    logits: (batch_size, seq_len, vocab_size)
    target: (batch_size, seq_len)
    """
    vocab_size = logits.size(-1)
    confidence = 1.0 - smoothing
    low_conf = smoothing / (vocab_size - 1)

    # Flatten
    logits = logits.view(-1, vocab_size)          # (batch*seq_len, vocab_size)
    target = target.view(-1)                      # (batch*seq_len)

    # Exclude pad tokens (pad_id=1)
    pad_mask = (target == 1)
    target[pad_mask] = 0  # temporarily to avoid out of index

    with torch.no_grad():
        true_dist = logits.new_zeros(logits.size())
        true_dist.fill_(low_conf)
        true_dist.scatter_(1, target.unsqueeze(1), confidence)

    log_probs = F.log_softmax(logits, dim=-1)

    # Zero out the loss for padded tokens
    loss = -(true_dist * log_probs).sum(dim=-1)
    loss = loss.masked_fill_(pad_mask, 0.0).mean()

    return loss
