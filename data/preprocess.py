"""
Preprocess (tokenize, apply SentencePiece BPE) the WMT14 (EN–DE) dataset.
Saves tokenized files in data/tokenized/.
"""

import os
import yaml
import sentencepiece as spm
from datasets import Dataset
from datasets import load_from_disk

def load_partitions(cfg):
    raw_dir = os.path.join("data", "raw")
    train_data = load_from_disk(os.path.join(raw_dir, "train.arrow"))
    valid_data = load_from_disk(os.path.join(raw_dir, "validation.arrow"))
    test_data = load_from_disk(os.path.join(raw_dir, "test.arrow"))
    return train_data, valid_data, test_data

def build_combined_corpus(train_data, src_lang, tgt_lang, combined_file):
    """
    Builds a combined text file from source and target fields for BPE training.
    The huggingface WMT14 dataset has the form:
        example["translation"]["en"]
        example["translation"]["de"]
    """
    count = 0
    with open(combined_file, "w", encoding="utf-8") as f:
        for example in train_data:
            # The text is in example["translation"][lang]
            if "translation" in example:
                translation_dict = example["translation"]
                # Make sure both langs exist
                if src_lang in translation_dict and tgt_lang in translation_dict:
                    src_text = translation_dict[src_lang].replace("\n", " ")
                    tgt_text = translation_dict[tgt_lang].replace("\n", " ")
                    f.write(src_text + "\n")
                    f.write(tgt_text + "\n")
                    count += 1
    print(f"Combined corpus lines written (each line is a single sentence): {count*2}")

def train_bpe_model(input_file, model_prefix, vocab_size=8000):
    """
    Train a SentencePiece BPE model on the combined corpus.
    """
    print(f"Training BPE with vocab size={vocab_size}...")
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        "--unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3 --user_defined_symbols=[SEP]"
    )
    print("BPE training complete!")

def apply_bpe_to_dataset(dataset: Dataset, sp, src_lang, tgt_lang, output_src_file, output_tgt_file):
    """
    Apply the trained BPE model to a dataset split, saving tokenized text files.
    """
    os.makedirs(os.path.dirname(output_src_file), exist_ok=True)
    line_count = 0

    with open(output_src_file, "w", encoding="utf-8") as sf, open(output_tgt_file, "w", encoding="utf-8") as tf:
        for example in dataset:
            if "translation" in example:
                translation_dict = example["translation"]
                if src_lang in translation_dict and tgt_lang in translation_dict:
                    src_line = translation_dict[src_lang].strip().replace("\n", " ")
                    tgt_line = translation_dict[tgt_lang].strip().replace("\n", " ")

                    src_tokens = sp.encode_as_pieces(src_line)
                    tgt_tokens = sp.encode_as_pieces(tgt_line)

                    sf.write(" ".join(src_tokens) + "\n")
                    tf.write(" ".join(tgt_tokens) + "\n")
                    line_count += 1

    print(f"Wrote {line_count} lines to {output_src_file} and {output_tgt_file}")

def main(config_path="./config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    src_lang = cfg["data"]["src_lang"]  # "en"
    tgt_lang = cfg["data"]["tgt_lang"]  # "de"
    vocab_size = cfg["data"]["vocab_size"]

    combined_corpus_file = cfg["data"]["combined_corpus_file"]
    vocab_prefix = cfg["data"]["vocab_prefix"]

    train_data, valid_data, test_data = load_partitions(cfg)

    print("Building combined corpus from training data...")
    build_combined_corpus(train_data, src_lang, tgt_lang, combined_corpus_file)

    print("Training BPE model...")
    train_bpe_model(combined_corpus_file, vocab_prefix, vocab_size=vocab_size)

    # Load trained SentencePiece model
    sp = spm.SentencePieceProcessor(model_file=f"{vocab_prefix}.model")

    print("Applying BPE to train set...")
    apply_bpe_to_dataset(
        train_data,
        sp,
        src_lang,
        tgt_lang,
        cfg["data"]["train_bpe_src"],
        cfg["data"]["train_bpe_tgt"]
    )

    print("Applying BPE to validation set...")
    apply_bpe_to_dataset(
        valid_data,
        sp,
        src_lang,
        tgt_lang,
        cfg["data"]["valid_bpe_src"],
        cfg["data"]["valid_bpe_tgt"]
    )

    print("Applying BPE to test set...")
    apply_bpe_to_dataset(
        test_data,
        sp,
        src_lang,
        tgt_lang,
        cfg["data"]["test_bpe_src"],
        cfg["data"]["test_bpe_tgt"]
    )

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
