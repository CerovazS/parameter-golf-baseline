# Developer Setup

## 1. Download Data

Data should be downloaded to **`$SCRATCH`**, which has large per-user storage and avoids filling up the shared `$WORK` quota (see [CLUSTER.md](CLUSTER.md#2-filesystems) for filesystem details). Note that `$SCRATCH` is auto-purged after ~30 days of inactivity.

Run the download from a **login node** (which has internet access):

```bash
# Full dataset (80 training shards, ~8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 --target $SCRATCH/data

# Smoke test with a single shard
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1 --target $SCRATCH/data
```

## 2. Environment Variables

Copy the template and fill in your values:

```bash
cp .env.template .env
```

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to the tokenized dataset directory, e.g. `$SCRATCH/data/datasets/fineweb10B_sp1024` |
| `TOKENIZER_PATH` | Path to the tokenizer model file, e.g. `$SCRATCH/data/tokenizers/fineweb_1024_bpe.model` |
| `PROXY_PORT` | Squid proxy port on `login01` for internet access from compute nodes |

Point `DATA_PATH` and `TOKENIZER_PATH` to the corresponding subdirectories under the `--target` used during download.
