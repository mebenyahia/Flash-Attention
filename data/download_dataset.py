"""
Script to download a reduced version of WMT14 (English-German) via Hugging Face Datasets.
Saves a small subset (e.g., 1%) for demonstration.
"""

import os
import yaml
from datasets import load_dataset

def download_reduced_wmt14(config_path="./config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["data"]["dataset_name"]    # "wmt14"
    dataset_config = cfg["data"]["dataset_config"]  # "de-en"
    subset_percentage = cfg["data"]["subset_percentage"]  # e.g. 1
    train_split = cfg["data"]["train_split"]  # "train"
    valid_split = cfg["data"]["valid_split"]  # "validation"
    test_split = cfg["data"]["test_split"]    # "test"

    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading a {subset_percentage}% subset of WMT14 {dataset_config}...")
    train_subset = f"{train_split}[:{subset_percentage}%]"
    valid_subset = valid_split
    test_subset = test_split

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split={
            "train": train_subset,
            "validation": valid_subset,
            "test": test_subset
        }
    )

    dataset["train"].save_to_disk(os.path.join(raw_dir, "train.arrow"))
    dataset["validation"].save_to_disk(os.path.join(raw_dir, "validation.arrow"))
    dataset["test"].save_to_disk(os.path.join(raw_dir, "test.arrow"))

    print("Download and subset extraction complete!")

if __name__ == "__main__":
    download_reduced_wmt14()
