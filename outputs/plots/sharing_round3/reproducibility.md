# Reproducibility

## Repository
- URL: https://github.com/CerovazS/parameter-golf (private)
- Branch: weight_sharing
- Commit SHA: 7082ab901c1975be58569d20f388aa75130a6216

## File Map
- `train_gpt_sharing.py`: Training script with shared low-rank factorization support. All Round 3 runs use this script.
- `tools/plot_sharing_round3.py`: Generates tradeoff frontier and BPB bar chart plots for Round 3 results.
- `outputs/report_sharing_experiments_round3.md`: Full written analysis of Round 3 results.

## Runs

All 13 Round 3 runs were submitted via `sbatch` on Leonardo CINECA. Each run used a
dedicated 4×A100 node with 700-second wall budget.

### SLURM job IDs and configs

| Run | Job ID | ro | rq | rmlp_fc | rmlp_proj | pairs |
|-----|--------|----|----|---------|-----------|-------|
| share4_o_2pairs | 39007754 | 256 | 0 | 0 | 0 | 2 |
| share4_o_3pairs | 39007755 | 256 | 0 | 0 | 0 | 3 |
| share4_o_4pairs | 39007756 | 256 | 0 | 0 | 0 | 4 |
| share4_q_1pair  | 39007757 | 0 | 256 | 0 | 0 | 1 |
| share4_q_2pairs | 39007758 | 0 | 256 | 0 | 0 | 2 |
| share4_mlp_fc_1p | 39007759 | 0 | 0 | 256 | 0 | 1 |
| share4_mlp_proj_1p | 39007760 | 0 | 0 | 0 | 256 | 1 |
| share4_mlp_both_1p | 39007761 | 0 | 0 | 256 | 256 | 1 |
| share4_mlp_fc_2p | 39007763 | 0 | 0 | 256 | 0 | 2 |
| share4_o_q_1p | 39007764 | 256 | 256 | 0 | 0 | 1 |
| share4_o_q_2p | 39007765 | 256 | 256 | 0 | 0 | 2 |
| share4_o_mlp_fc_1p | 39007766 | 256 | 0 | 256 | 0 | 1 |
| share4_full_1p | 39007767 | 256 | 256 | 256 | 256 | 1 |

### Example launch command (Q p=1)

```bash
SHARING_RANK_Q=256 SHARING_NUM_PAIRS=1 \
MAX_WALLCLOCK_SECONDS=700 SEED=1337 \
torchrun --standalone --nproc_per_node=4 train_gpt_sharing.py
```

All other runs follow the same pattern substituting the relevant `SHARING_RANK_*` and
`SHARING_NUM_PAIRS` values from the table above.

## Environment Setup

```bash
module load cuda/12.2
cd /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
source .venv/bin/activate
# Proxy (required on compute nodes):
export http_proxy='http://login01:3128'
export https_proxy='http://login01:3128'
```

## Hardware
- Cluster: Leonardo CINECA
- Partition: boost_usr_prod
- Account: IscrC_YENDRI
- GPUs: 4× NVIDIA A100-SXM-64GB (64 GB HBM2e)
- CPUs: 32 per node
- Wall budget: 700 seconds per run

## Hyperparameters (all runs)
- Seed: 1337
- model_dim: 512
- num_heads: 8
- num_kv_heads: 4
- mlp_mult: 2
- vocab_size: 1024 (SentencePiece sp1024)
- num_layers: 9
- tied_embed: True
- MATRIX_LR: 0.04
- MUON_MOMENTUM: 0.95
- batch_size: 131072 tokens/GPU → 524288 tokens/step
- seq_len: 1024
- SHARING_RANK_V: 0 (all runs)
- ITERATIONS: 20000 (wall budget terminates earlier)
