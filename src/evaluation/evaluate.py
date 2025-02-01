import os
import json
import yaml
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sentencepiece as spm
import sacrebleu

from ..model.transformer import Transformer
from ..training.train import TranslationDataset, collate_fn
from .inference import generate_translation

############################################################
# 1) Metric Helpers
############################################################

def sentence_bleu_score(reference_tokens, hypothesis_tokens):
    """
    Computes sentence-level BLEU for a single reference-hypothesis pair using sacrebleu.
    """
    ref_str = " ".join(reference_tokens)
    hyp_str = " ".join(hypothesis_tokens)
    return sacrebleu.sentence_bleu(hyp_str, [ref_str]).score

def sentence_chrf_score(reference_tokens, hypothesis_tokens):
    """
    Computes sentence-level chrF (character n-gram F-score) using sacrebleu.
    """
    ref_str = " ".join(reference_tokens)
    hyp_str = " ".join(hypothesis_tokens)
    # chrF can be computed at sentence level too
    return sacrebleu.sentence_chrf(hyp_str, [ref_str]).score

def corpus_bleu_score(all_references, all_hypotheses):
    """
    Computes corpus-level BLEU via sacrebleu.
    """
    ref_strs = [" ".join(ref) for ref in all_references]
    hyp_strs = [" ".join(hyp) for hyp in all_hypotheses]
    bleu = sacrebleu.corpus_bleu(hyp_strs, [ref_strs])
    return bleu.score

def corpus_chrf_score(all_references, all_hypotheses):
    """
    Computes corpus-level chrF via sacrebleu.
    """
    ref_strs = [" ".join(ref) for ref in all_references]
    hyp_strs = [" ".join(hyp) for hyp in all_hypotheses]
    chrf = sacrebleu.corpus_chrf(hyp_strs, [ref_strs])
    return chrf.score

############################################################
# 2) Main Evaluation
############################################################

def evaluate_model(config_path="./config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    sp_model = spm.SentencePieceProcessor(model_file=f"{cfg['data']['vocab_prefix']}.model")

    # 2.1) Load test dataset
    test_dataset = TranslationDataset(cfg["data"]["test_bpe_src"], cfg["data"]["test_bpe_tgt"])
    
    # 2.2) Randomly select 20% of the test set
    test_size = len(test_dataset)
    subset_size = int(test_size * 0.2)
    subset_indices = random.sample(range(test_size), subset_size)
    test_subset = Subset(test_dataset, subset_indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, sp_model),
        drop_last=False
    )

    # 2.3) Build model
    src_vocab_size = sp_model.vocab_size()
    tgt_vocab_size = sp_model.vocab_size()
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        d_ff=cfg["model"]["d_ff"],
        max_len=cfg["model"]["max_position_embeddings"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    # 2.4) Load checkpoint
    last_epoch_ckpt = f"model_epoch_{cfg['training']['max_epochs']}.pt"
    ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], last_epoch_ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 2.5) Data structures for metrics
    global_references = []
    global_hypotheses = []

    # For plotting, track metrics per batch
    batch_sentence_bleu_scores = []
    batch_sentence_chrf_scores = []
    batch_corpus_bleu_scores = []
    batch_corpus_chrf_scores = []

    # For histograms of sentence-level BLEU & chrF across entire subset
    sentence_bleu_distribution = []
    sentence_chrf_distribution = []

    ############################################################
    # 2.6) Evaluation Loop
    ############################################################
    with torch.no_grad():
        for src, tgt_in, tgt_out in tqdm(test_loader, desc="Evaluating"):
            src = src.to(device)

            # Generate translations
            hyp_ids_batch = generate_translation(model, src, sp_model, cfg)

            # Convert references => token lists
            batch_ref_tokens = []
            for ref_seq in tgt_out:
                ref_seq = ref_seq.tolist()
                if 1 in ref_seq:
                    pad_idx = ref_seq.index(1)
                else:
                    pad_idx = len(ref_seq)
                ref_seq = ref_seq[:pad_idx]  # remove PAD
                filtered = [t for t in ref_seq if t not in [1,2,3]]  # remove BOS=2, EOS=3, PAD=1
                ref_text = sp_model.DecodeIds(filtered)
                batch_ref_tokens.append(ref_text.split())

            # Convert hypotheses => token lists
            batch_hyp_tokens = []
            for h in hyp_ids_batch:
                filtered_h = [tid for tid in h if tid not in [1,2,3]]
                hyp_text = sp_model.DecodeIds(filtered_h)
                batch_hyp_tokens.append(hyp_text.split())

            # 2.6.1) Compute sentence-level metrics for each sample in batch
            sentence_bleus = []
            sentence_chrfs = []
            for ref_toks, hyp_toks in zip(batch_ref_tokens, batch_hyp_tokens):
                sb = sentence_bleu_score(ref_toks, hyp_toks)
                sc = sentence_chrf_score(ref_toks, hyp_toks)
                sentence_bleus.append(sb)
                sentence_chrfs.append(sc)
            
            # Store for histogram over entire subset
            sentence_bleu_distribution.extend(sentence_bleus)
            sentence_chrf_distribution.extend(sentence_chrfs)

            # Average these metrics for the batch
            avg_bleu_batch = sum(sentence_bleus) / len(sentence_bleus) if sentence_bleus else 0.0
            avg_chrf_batch = sum(sentence_chrfs) / len(sentence_chrfs) if sentence_chrfs else 0.0

            # 2.6.2) Update global references/hypotheses for corpus-level metrics
            global_references.extend(batch_ref_tokens)
            global_hypotheses.extend(batch_hyp_tokens)

            # 2.6.3) Compute corpus-level metrics so far
            corp_bleu = corpus_bleu_score(global_references, global_hypotheses)
            corp_chrf = corpus_chrf_score(global_references, global_hypotheses)

            # 2.6.4) Record for batch-level plots
            batch_sentence_bleu_scores.append(avg_bleu_batch)
            batch_sentence_chrf_scores.append(avg_chrf_batch)
            batch_corpus_bleu_scores.append(corp_bleu)
            batch_corpus_chrf_scores.append(corp_chrf)

    ############################################################
    # 2.7) Final Metrics Over the Entire 20% Subset
    ############################################################
    final_bleu = corpus_bleu_score(global_references, global_hypotheses)
    final_chrf = corpus_chrf_score(global_references, global_hypotheses)

    avg_sentence_bleu_all = sum(batch_sentence_bleu_scores) / len(batch_sentence_bleu_scores)
    avg_sentence_chrf_all = sum(batch_sentence_chrf_scores) / len(batch_sentence_chrf_scores)

    print(f"\nFinal Corpus BLEU on 20% subset: {final_bleu:.2f}")
    print(f"Final Corpus chrF on 20% subset: {final_chrf:.2f}")
    print(f"Average Sentence-Level BLEU (batch-averaged): {avg_sentence_bleu_all:.2f}")
    print(f"Average Sentence-Level chrF (batch-averaged): {avg_sentence_chrf_all:.2f}")

    ############################################################
    # 2.8) Visualizations
    ############################################################

    # (A) Line Plots of Batch Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sentence_bleu_scores, label="Batch-Averaged Sentence BLEU")
    plt.plot(batch_corpus_bleu_scores, label="Cumulative Corpus BLEU")
    plt.plot(batch_sentence_chrf_scores, label="Batch-Averaged Sentence chrF")
    plt.plot(batch_corpus_chrf_scores, label="Cumulative Corpus chrF")
    plt.xlabel("Batch Index")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics over Batches (20% of Test)")
    plt.legend()
    plot_line_save_path = "batch_metrics_plot.png"
    plt.savefig(plot_line_save_path)
    print(f"Saved batch metrics line plot -> {plot_line_save_path}")
    plt.show()

    # (B) Histogram of Sentence-Level BLEU
    plt.figure(figsize=(8, 5))
    plt.hist(sentence_bleu_distribution, bins=30, color='blue', alpha=0.7)
    plt.title("Distribution of Sentence-Level BLEU (20% subset)")
    plt.xlabel("Sentence BLEU")
    plt.ylabel("Count")
    plt.grid(True)
    hist_bleu_path = "hist_sentence_bleu.png"
    plt.savefig(hist_bleu_path)
    print(f"Saved sentence BLEU histogram -> {hist_bleu_path}")
    plt.show()

    # (C) Histogram of Sentence-Level chrF
    plt.figure(figsize=(8, 5))
    plt.hist(sentence_chrf_distribution, bins=30, color='green', alpha=0.7)
    plt.title("Distribution of Sentence-Level chrF (20% subset)")
    plt.xlabel("Sentence chrF")
    plt.ylabel("Count")
    plt.grid(True)
    hist_chrf_path = "hist_sentence_chrf.png"
    plt.savefig(hist_chrf_path)
    print(f"Saved sentence chrF histogram -> {hist_chrf_path}")
    plt.show()

    ############################################################
    # 2.9) Save Results to JSON
    ############################################################
    results_dict = {
        "subset_size": subset_size,
        "final_corpus_bleu": final_bleu,
        "final_corpus_chrf": final_chrf,
        "avg_sentence_bleu_all": avg_sentence_bleu_all,
        "avg_sentence_chrf_all": avg_sentence_chrf_all,
        "batch_sentence_bleu_scores": batch_sentence_bleu_scores,
        "batch_corpus_bleu_scores": batch_corpus_bleu_scores,
        "batch_sentence_chrf_scores": batch_sentence_chrf_scores,
        "batch_corpus_chrf_scores": batch_corpus_chrf_scores,
        "sentence_bleu_distribution": sentence_bleu_distribution,
        "sentence_chrf_distribution": sentence_chrf_distribution
    }

    json_save_path = "evaluation_metrics.json"
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Saved evaluation results to: {json_save_path}")

if __name__ == "__main__":
    evaluate_model()

############################################################