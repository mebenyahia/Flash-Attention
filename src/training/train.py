import os
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# For PyTorch 2.0+ AMP usage
from torch.amp import autocast, GradScaler

from ..model.transformer import Transformer
from .optimizer import create_optimizer_and_scheduler
from .regularization import label_smoothing_loss

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        self.src_lines = open(src_file, "r", encoding="utf-8").read().strip().splitlines()
        self.tgt_lines = open(tgt_file, "r", encoding="utf-8").read().strip().splitlines()
        assert len(self.src_lines) == len(self.tgt_lines), (
            "Source and target files must have the same number of lines."
        )

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = self.src_lines[idx].split()
        tgt_tokens = self.tgt_lines[idx].split()
        return src_tokens, tgt_tokens

def collate_fn(batch, sp_model, pad_id=1):
    src_batch, tgt_batch = zip(*batch)
    src_ids = [sp_model.EncodeAsIds(" ".join(x)) for x in src_batch]
    tgt_ids = [sp_model.EncodeAsIds(" ".join(x)) for x in tgt_batch]

    max_src_len = max(len(s) for s in src_ids)
    # +1 to accommodate BOS/EOS
    max_tgt_len = max(len(t) for t in tgt_ids) + 1

    src_tensor = []
    tgt_in_tensor = []
    tgt_out_tensor = []

    bos_id = 2
    eos_id = 3

    for s, t in zip(src_ids, tgt_ids):
        s_pad = s + [pad_id]*(max_src_len - len(s))
        t_in = [bos_id] + t
        t_out = t + [eos_id]

        t_in_pad = t_in + [pad_id]*(max_tgt_len - len(t_in))
        t_out_pad = t_out + [pad_id]*(max_tgt_len - len(t_out))

        src_tensor.append(s_pad)
        tgt_in_tensor.append(t_in_pad)
        tgt_out_tensor.append(t_out_pad)

    return (
        torch.tensor(src_tensor, dtype=torch.long),
        torch.tensor(tgt_in_tensor, dtype=torch.long),
        torch.tensor(tgt_out_tensor, dtype=torch.long),
    )

def train_model(config_path="./config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor(model_file=f"{cfg['data']['vocab_prefix']}.model")

    device = cfg["training"]["device"]
    batch_size = cfg["training"]["batch_size"]
    gradient_accum_steps = cfg["training"]["gradient_accumulation_steps"]
    max_epochs = cfg["training"]["max_epochs"]
    log_interval = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_checkpoint_interval"]

    train_dataset = TranslationDataset(cfg["data"]["train_bpe_src"], cfg["data"]["train_bpe_tgt"])
    valid_dataset = TranslationDataset(cfg["data"]["valid_bpe_src"], cfg["data"]["valid_bpe_tgt"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, sp_model)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, sp_model)
    )

    src_vocab_size = sp_model.vocab_size()
    tgt_vocab_size = sp_model.vocab_size()

    d_model = cfg["model"]["d_model"]
    n_heads = cfg["model"]["n_heads"]
    num_encoder_layers = cfg["model"]["num_encoder_layers"]
    num_decoder_layers = cfg["model"]["num_decoder_layers"]
    d_ff = cfg["model"]["d_ff"]
    dropout = cfg["model"]["dropout"]
    max_len = cfg["model"]["max_position_embeddings"]

    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        max_len,
        dropout
    ).to(device)

    optimizer, scheduler = create_optimizer_and_scheduler(model, cfg)

    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])

    global_step = 0
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for step, (src, tgt_in, tgt_out) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            with autocast("cuda", enabled=cfg["training"]["mixed_precision"]):
                logits = model(src, tgt_in)
                loss = label_smoothing_loss(logits, tgt_out, cfg["model"]["label_smoothing"])

            scaler.scale(loss).backward()
            if (step + 1) % gradient_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            total_loss += loss.item()

            if global_step % log_interval == 0 and global_step > 0:
                avg_loss = total_loss / (step + 1)
                print(f"Step {global_step}: loss = {avg_loss:.4f}")

            if global_step % save_interval == 0 and global_step > 0:
                ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], f"model_step_{global_step}.pt")
                os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved -> {ckpt_path}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average training loss: {avg_loss:.4f}")

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for src, tgt_in, tgt_out in valid_loader:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                with autocast("cuda", enabled=cfg["training"]["mixed_precision"]):
                    logits = model(src, tgt_in)
                    loss = label_smoothing_loss(logits, tgt_out, cfg["model"]["label_smoothing"])
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")

        ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved -> {ckpt_path}")

if __name__ == "__main__":
    train_model()
