#!/usr/bin/env bash
set -e

CONFIG_PATH="config/config.yaml"
python -m src.training.train --config_path $CONFIG_PATH
