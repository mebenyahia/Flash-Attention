import pytest
from src.evaluation.metrics import compute_bleu

def test_compute_bleu():
    references = [
        [["this", "is", "a", "test"]],
        [["another", "sentence"]]
    ]
    hypotheses = [
        ["this", "is", "a", "test"],
        ["another", "phrase"]
    ]
    score = compute_bleu(references, hypotheses)
    assert 0 <= score <= 100, "BLEU score must be between 0 and 100"
