import torch
import torch.nn.functional as F
# For inference AMP
from torch.amp import autocast

def generate_translation(model, src, sp_model, cfg):
    """
    Generates translations for a batch of source sentences (src).
    We'll use beam search or greedy, but must use autocast for fp16.
    """
    beam_size = cfg["beam_search"]["beam_size"]
    max_length = cfg["beam_search"]["max_length"]

    if beam_size == 1:
        return greedy_search(model, src, sp_model, max_length)
    else:
        return beam_search(model, src, sp_model, beam_size, max_length)

def greedy_search(model, src, sp_model, max_length):
    model.eval()
    with torch.no_grad(), autocast("cuda", enabled=True, dtype=torch.float16):
        bsz = src.size(0)
        device = src.device
        bos_id = 2
        eos_id = 3
        pad_id = 1

        # Encode
        enc_out = model.src_embedding(src)
        enc_out = model.src_pos_encoding(enc_out)
        enc_out = model.encoder(enc_out)

        # Initialize target
        tgt_seq = torch.LongTensor([bos_id]*bsz).unsqueeze(1).to(device)

        for _ in range(max_length):
            dec_inp = model.tgt_embedding(tgt_seq)
            dec_inp = model.tgt_pos_encoding(dec_inp)
            dec_out = model.decoder(dec_inp, enc_out)
            logits = model.output_proj(dec_out)  # (bsz, seq_len, vocab_size)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tgt_seq = torch.cat([tgt_seq, next_token], dim=1)

        # Convert to list
        hyp_ids = tgt_seq.tolist()
        # Remove initial BOS
        return [seq[1:] for seq in hyp_ids]

def beam_search(model, src, sp_model, beam_size, max_length):
    model.eval()
    with torch.no_grad(), autocast("cuda", enabled=True, dtype=torch.float16):
        bsz = src.size(0)
        device = src.device
        bos_id = 2
        eos_id = 3
        pad_id = 1

        # Encode
        enc_out = model.src_embedding(src)
        enc_out = model.src_pos_encoding(enc_out)
        enc_out = model.encoder(enc_out)

        # We'll do single-sentence approach per batch item
        all_hypotheses = []
        for b in range(bsz):
            enc_out_b = enc_out[b:b+1]  # shape [1, src_len, d_model]
            beams = [([bos_id], 0.0)]   # (tokens, score)

            for _ in range(max_length):
                new_beams = []
                for seq, score in beams:
                    if seq[-1] == eos_id:
                        new_beams.append((seq, score))
                        continue

                    # Prepare decoder input
                    tgt_seq = torch.LongTensor(seq).unsqueeze(0).to(device)
                    dec_inp = model.tgt_embedding(tgt_seq)
                    dec_inp = model.tgt_pos_encoding(dec_inp)
                    dec_out = model.decoder(dec_inp, enc_out_b)
                    logits = model.output_proj(dec_out)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                    topk_vals, topk_indices = log_probs.topk(beam_size)
                    for val, idx in zip(topk_vals, topk_indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + val.item()
                        new_beams.append((new_seq, new_score))

                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]

                # If all beams end with EOS, break
                if all(bm[0][-1] == eos_id for bm in beams):
                    break

            best_seq, best_score = sorted(beams, key=lambda x: x[1], reverse=True)[0]
            all_hypotheses.append(best_seq)

        return all_hypotheses
