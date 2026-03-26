#!/bin/bash
# Download the FineWeb dataset with the sp1024 tokenizer.
# Run this on a login node (has internet) before submitting training jobs.
#
# Usage:
#   bash scripts/download_data.sh            # full dataset (80 shards, ~8B tokens)
#   bash scripts/download_data.sh 1          # smoke test (1 shard)

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_SHARDS="${1:-80}"

echo "Downloading FineWeb sp1024 dataset (${TRAIN_SHARDS} training shards)..."
uv run python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"

echo "Done. Data in ./data/datasets/fineweb10B_sp1024/"
echo "Tokenizer in ./data/tokenizers/"
