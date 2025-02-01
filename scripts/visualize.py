#!/usr/bin/env python3
"""
A unified visualization script for your flash-transformer repository.

It supports four modes:
  1. embeddings:  Plot the model's source embeddings matrix
  2. curves:      Plot training/validation loss curves from a CSV
  3. sample:      Generate translations for a small subset of the test set
  4. flash_out:   Demonstrate hooking into FlashAttention to capture 'output' 
                  (NOT the standard attention weights, which flash-attn doesn't store)

Usage:
  python scripts/visualize.py --mode embeddings
  python scripts/visualize.py --mode curves
  python scripts/visualize.py --mode sample
  python scripts/visualize.py --mode flash_out
"""

import os
import yaml
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast

# Import your modules
from src.visualization.embedding_visuals import plot_embeddings
from src.visualization.training_curves import plot_training_curves
from src.visualization.plot_attention import plot_attention

from src.training.train import TranslationDataset, collate_fn
from src.evaluation.inference import generate_translation
from src.evaluation.metrics import compute_bleu
from src.model.transformer import Transformer
from src.model.flash_attention import FlashAttention
from src.model.encoder import EncoderLayer
from src.model.decoder import DecoderLayer

import sentencepiece as spm

##############################################
# GLOBALS & HOOK HANDLING
##############################################
captured_output = None

def flash_attn_hook(module, input, output):
    """
    A forward hook that captures the output of FlashAttention for demonstration.
    NOTE: This is NOT the typical [tgt_len, src_len] attention matrix, 
    but the final [B, S, n_heads, head_dim] result after softmax * V.
    """
    global captured_output
    captured_output = output  # shape [B, S, n_heads, head_dim]

##############################################
# EMBEDDING VISUALIZATION
##############################################
def visualize_embeddings(config_path):
    """
    Load your model checkpoint, then plot the source embedding matrix.
    """
    print("Loading config...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    sp_model = spm.SentencePieceProcessor(model_file=f"{cfg['data']['vocab_prefix']}.model")

    # Build model
    src_vocab_size = sp_model.vocab_size()
    tgt_vocab_size = sp_model.vocab_size()
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        cfg["model"]["d_model"],
        cfg["model"]["n_heads"],
        cfg["model"]["num_encoder_layers"],
        cfg["model"]["num_decoder_layers"],
        cfg["model"]["d_ff"],
        cfg["model"]["max_position_embeddings"],
        cfg["model"]["dropout"]
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"],
                             f"model_epoch_{cfg['training']['max_epochs']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Grab the source embedding weight: shape [vocab_size, d_model]
    emb_matrix = model.src_embedding.embedding.weight.data.cpu()

    print(f"Plotting embeddings with shape: {emb_matrix.shape}")
    plot_embeddings(emb_matrix, title="Source Embeddings")


##############################################
# TRAINING CURVES
##############################################
def visualize_curves(log_csv="train_log.csv"):
    """
    Reads a CSV file that contains columns: step,train_loss,valid_loss
    Then calls plot_training_curves.
    """
    if not os.path.isfile(log_csv):
        raise FileNotFoundError(f"No training log CSV found at {log_csv}")

    import csv
    steps = []
    train_losses = []
    valid_losses = []

    with open(log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            valid_losses.append(float(row["valid_loss"]))

    print(f"Loaded {len(steps)} steps from {log_csv}.")
    plot_training_curves(train_losses, valid_losses)


##############################################
# SAMPLE TRANSLATIONS
##############################################
def visualize_sample_translations(config_path, num_samples=5):
    """
    Load a small subset (num_samples) of the test set, run actual inference with generate_translation,
    and print source/target/hypothesis. This doesn't produce an attention heatmap but
    shows real translations for debugging BLEU or the model's output quality.
    """
    print("Loading config...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    sp_model = spm.SentencePieceProcessor(model_file=f"{cfg['data']['vocab_prefix']}.model")

    # Build model
    src_vocab_size = sp_model.vocab_size()
    tgt_vocab_size = sp_model.vocab_size()
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        cfg["model"]["d_model"],
        cfg["model"]["n_heads"],
        cfg["model"]["num_encoder_layers"],
        cfg["model"]["num_decoder_layers"],
        cfg["model"]["d_ff"],
        cfg["model"]["max_position_embeddings"],
        cfg["model"]["dropout"]
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"],
                             f"model_epoch_{cfg['training']['max_epochs']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Load test dataset
    test_src = cfg["data"]["test_bpe_src"]
    test_tgt = cfg["data"]["test_bpe_tgt"]
    test_dataset = TranslationDataset(test_src, test_tgt)

    # Create a small DataLoader
    sub_dataset = [test_dataset[i] for i in range(min(num_samples, len(test_dataset)))]
    test_loader = DataLoader(
        sub_dataset,
        batch_size=1,  # do one at a time for clarity
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, sp_model),
        drop_last=False
    )

    # Inference
    with torch.no_grad():
        for i, (src, tgt_in, tgt_out) in enumerate(test_loader):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            hyp_ids = generate_translation(model, src, sp_model, cfg)

            # Convert the source IDs back to text
            # src is shape [1, src_len]
            src_line = src[0].tolist()
            if 1 in src_line:  # pad_id=1
                src_line = src_line[:src_line.index(1)]
            src_text = sp_model.DecodeIds([s for s in src_line if s not in [1,2,3]])

            # Convert the reference IDs
            ref_line = tgt_out[0].tolist()
            if 1 in ref_line:
                ref_line = ref_line[:ref_line.index(1)]
            ref_text = sp_model.DecodeIds([t for t in ref_line if t not in [1,2,3]])

            # Convert the hypothesis IDs
            # hyp_ids is a list of lists, one for each batch element
            hyp_line = hyp_ids[0]
            hyp_text = sp_model.DecodeIds([h for h in hyp_line if h not in [1,2,3]])

            print(f"\nExample {i+1}:")
            print(f"SRC: {src_text}")
            print(f"REF: {ref_text}")
            print(f"HYP: {hyp_text}")


##############################################
# CAPTURE FLASHATTN OUTPUT (NOT ATTENTION WEIGHTS)
##############################################
def visualize_flash_out(config_path, sample_text="Hello world!"):
    """
    Demonstrates hooking the FlashAttention forward pass to capture the final 
    [B,S,n_heads,head_dim] output, which is NOT a typical [tgt_len, src_len] weight matrix.

    We'll do a minimal forward pass with a dummy target to trigger the encoder's self-attn.
    Then we plot the norm of that tensor as a 'pseudo-attention map' 
    (this is purely a demonstration).

    If you want actual attention weights, you'd have to rewrite or modify flash-attn for that.
    """
    global captured_output
    captured_output = None

    print("Loading config...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    sp_model = spm.SentencePieceProcessor(model_file=f"{cfg['data']['vocab_prefix']}.model")

    # Build model
    src_vocab_size = sp_model.vocab_size()
    tgt_vocab_size = sp_model.vocab_size()
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        cfg["model"]["d_model"],
        cfg["model"]["n_heads"],
        cfg["model"]["num_encoder_layers"],
        cfg["model"]["num_decoder_layers"],
        cfg["model"]["d_ff"],
        cfg["model"]["max_position_embeddings"],
        cfg["model"]["dropout"]
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"],
                             f"model_epoch_{cfg['training']['max_epochs']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # We'll hook, e.g., the first encoder layer's self_attn
    first_encoder_layer = model.encoder.layers[0]
    hook_handle = first_encoder_layer.self_attn.register_forward_hook(flash_attn_hook)

    # Create a source input from sample_text
    src_ids = sp_model.EncodeAsIds(sample_text)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    # Minimal target so forward() doesn't fail
    # For instance, shape [1,1] with BOS=2
    bos_id = 2
    tgt_tensor = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    # Forward pass in half precision if needed
    from torch.amp import autocast
    with torch.no_grad(), autocast("cuda", enabled=cfg["training"]["mixed_precision"]):
        _ = model(src_tensor, tgt_tensor)

    hook_handle.remove()  # remove the hook

    if captured_output is None:
        print("No output was captured from FlashAttention. Possibly the hook didn't fire.")
        return

    # captured_output: [B, S, n_heads, head_dim]
    print("Captured FlashAttention output shape:", captured_output.shape)

    # For a pseudo-visualization: take L2 norm across head_dim => [B, S, n_heads]
    # Then pick the first example: [S, n_heads]
    flash_mag = captured_output.norm(dim=-1)[0]  # shape [S, n_heads]
    # We might want shape [n_heads, S] for the heatmap
    flash_mag = flash_mag.transpose(0,1)  # => [n_heads, S]

    # We'll label heads on the y-axis, positions on the x-axis
    # The "positions" can be the tokens, so let's decode them
    tokens = sp_model.IdToPiece(src_ids[0])  # just the first token
    # Actually we need the entire src_ids
    src_pieces = [sp_model.IdToPiece(id_) for id_ in src_ids]

    print(f"Plotting the norm of FlashAttention's output for each head vs. position.")
    plot_attention(
        attn_weights=flash_mag,  # shape [n_heads, seq_len]
        src_tokens=src_pieces,
        tgt_tokens=[f"H{i}" for i in range(cfg['model']['n_heads'])]
    )

##############################################
# MAIN
##############################################
def main():
    parser = argparse.ArgumentParser("Visualization script for flash-transformer")
    parser.add_argument("--mode", type=str, default="embeddings",
                        help="Which visualization to run: embeddings | curves | sample | flash_out")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--log_csv", type=str, default="train_log.csv",
                        help="Path to training log CSV for curves mode")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of test samples for sample mode")
    parser.add_argument("--sample_text", type=str, default="Hello world!",
                        help="Sample text for flash_out mode")
    args = parser.parse_args()

    if args.mode == "embeddings":
        visualize_embeddings(args.config)
    elif args.mode == "curves":
        visualize_curves(args.log_csv)
    elif args.mode == "sample":
        visualize_sample_translations(args.config, args.num_samples)
    elif args.mode == "flash_out":
        visualize_flash_out(args.config, args.sample_text)
    else:
        print("Unknown mode. Choose from: embeddings | curves | sample | flash_out.")

if __name__ == "__main__":
    main()
