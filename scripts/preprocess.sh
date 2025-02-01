#!/usr/bin/env bash
set -e

echo "Downloading dataset..."
python data/download_dataset.py

echo "Preprocessing dataset..."
python data/preprocess.py
