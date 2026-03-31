# Reproducibility Guide — Triton Newton-Schulz Kernel Experiments

## Overview

These experiments benchmark custom Triton CUDA kernels as drop-in replacements for the
Newton-Schulz orthogonalization steps inside the Muon optimizer used in `train_gpt.py`.
The Newton-Schulz loop contains two matrix-multiply-heavy operations:

1. `X @ X.T` — symmetric outer product used to compute `A = X @ X.T`
2. `b*A + c*(A @ A)` — fused polynomial update

The hypothesis is that hand-written Triton kernels for these operations — exploiting
symmetry and reducing memory traffic — can lower mean step time without affecting
convergence or final BPB.

Three iterative variants are compared against a control that uses `torch.compile` on
the unmodified Newton-Schulz implementation:

| Variant     | XXT kernel | ba\_plus\_cAA kernel |
|-------------|-----------|----------------------|
| Control     | no        | no (`torch.compile`) |
| Iter-1      | yes       | no                   |
| Iter-2      | yes       | yes                  |
| Iter-3      | no        | yes                  |

---

## Repository

- **Branch**: `kernels`
- **Commit**: `994f7cb0b2f348ddb708190b4c7f3818f2eda551`
- **Cluster**: Leonardo (CINECA), partition `boost_usr_prod`, account `IscrC_YENDRI`

---

## File Map

| File | Description |
|------|-------------|
| `train_gpt.py` | Iter-2 training script (XXT + ba\_plus\_cAA Triton kernels active) |
| `train_gpt_iter1.py` | Iter-1 training script (XXT Triton kernel only) |
| `train_gpt_iter3.py` | Iter-3 training script (ba\_plus\_cAA Triton kernel only; standard PyTorch X@X.T) |
| `train_gpt_main_ctrl.py` | Control script (original Newton-Schulz, `torch.compile`, no Triton) |
| `triton_kernels.py` | All Triton kernel implementations: `xxt_kernel`, `xtx_kernel`, `ba_plus_cAA_kernel`, and helpers |
| `scripts/bench_4xa100_control.sh` | SLURM batch script for the control 4×A100 run |
| `scripts/bench_4xa100_iter1.sh` | SLURM batch script for the Iter-1 4×A100 run |
| `scripts/bench_4xa100_iter2.sh` | SLURM batch script for the Iter-2 4×A100 run |
| `scripts/bench_4xa100_iter3.sh` | SLURM batch script for the Iter-3 4×A100 run |
| `tools/launch_run.sh` | Mandatory run wrapper: creates a unique run directory, snapshots code, writes `meta.json`, tees logs |
| `outputs/runs/kernels/` | Archived run artifacts for all four variants (`.pt` float checkpoints excluded) |
| `reports/triton_kernels_report_en.md` | Full technical report with analysis and plots |
| `reports/kernel_benchmark_4xA100.png` | Summary benchmark bar plot |

---

## How to Reproduce

### Prerequisites

```bash
cd /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
module load cuda/12.2

# Ensure the dataset is present (80 training shards, ~8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

All four SLURM scripts call `tools/launch_run.sh` internally, so a unique run
directory is created automatically and `meta.json` is finalized after training.

### Run Control (no Triton)

```bash
HYPOTHESIS="Control: torch.compile on Newton-Schulz, no Triton" \
  sbatch scripts/bench_4xa100_control.sh
```

### Run Iter-1 (XXT only)

```bash
HYPOTHESIS="Iter-1: XXT Triton kernel only in Newton-Schulz" \
  sbatch scripts/bench_4xa100_iter1.sh
```

### Run Iter-2 (XXT + ba\_plus\_cAA) — best performer

```bash
HYPOTHESIS="Iter-2: XXT + ba_plus_cAA Triton kernels in Newton-Schulz" \
  sbatch scripts/bench_4xa100_iter2.sh
```

### Run Iter-3 (ba\_plus\_cAA only)

```bash
HYPOTHESIS="Iter-3: ba_plus_cAA only Triton kernel in Newton-Schulz" \
  sbatch scripts/bench_4xa100_iter3.sh
```

---

## Key Results (4×A100, 2000 steps, SEED=1337)

All four 4×A100 runs share identical hyperparameters (2000 iterations,
`TRAIN_BATCH_TOKENS=524288`, `TRAIN_SEQ_LEN=1024`). Iter-1 data comes from
SLURM wall-clock logs only (see note below).

| Variant                   | Script                   | mean\_step\_ms | val\_bpb | Δ step vs ctrl | Δ bpb vs ctrl |
|---------------------------|--------------------------|---------------:|----------:|---------------:|---------------:|
| Control (torch.compile)   | `train_gpt_main_ctrl.py` |         173.80 |   1.3610  |      —         |      —         |
| Iter-1 (XXT only)         | `train_gpt_iter1.py`     |         194.22 |   1.3608  |     −11.8 %    |     −0.0002    |
| **Iter-2 (XXT + ba\_plus\_cAA)** | `train_gpt.py`  |     **158.86** | **1.3599**|    **+8.6 %**  |   **−0.0011**  |
| Iter-3 (ba\_plus\_cAA only) | `train_gpt_iter3.py`   |         159.16 |   1.3629  |      +8.5 %    |     +0.0019    |

Sign convention: a positive Δ step means faster (fewer ms per step); a negative
Δ bpb means better compression.

**Reading**: Iter-2 is both the fastest and the most accurate variant.
Iter-1 alone is slower than the control (XXT in isolation does not win back the
`torch.compile` overhead). Iter-3 is fast but slightly worse in BPB than
the control, suggesting that fusing `b*A + c*A@A` without the symmetric X@X.T
trick recovers speed but loses a small regularizing effect from the compiled path.

Exact result files per run are at:

```
outputs/runs/kernels/ctrl_4xa100/meta.json
outputs/runs/kernels/iter2_xxt_ba_4xa100/meta.json
outputs/runs/kernels/iter3_ba_only_4xa100/meta.json
outputs/runs/kernels/iter1_xxt_only_slurm.out   # raw log only
```

---

## Notes on Iter-1 Data

The Iter-1 run directory was lost due to a timestamp collision: two SLURM jobs were
submitted at the same second and `tools/launch_run.sh` generated the same
`run_20260330_174215` directory name for both. The Iter-1 job wrote into the
directory first and was then silently overwritten by the control job when it also
attempted to finalize `meta.json`.

As a result:

- No `meta.json`, `output.log`, or `.ptz` artifact exists for Iter-1 as a standalone run.
- The only surviving evidence is the raw SLURM wall-clock log, archived at
  `outputs/runs/kernels/iter1_xxt_only_slurm.out` and `.err`.
- The final metrics (`val_bpb=1.3608`, `mean_step_ms≈194.22`) are extracted
  directly from that SLURM log.

---

## Environment

| Component | Version / source |
|-----------|-----------------|
| Python    | Managed by `uv` — see `pyproject.toml` |
| CUDA      | 12.2 (loaded via `module load cuda/12.2`) |
| PyTorch   | See `uv.lock` |
| Triton    | Bundled with PyTorch (no separate install) |
| Cluster   | Leonardo HPC (CINECA), A100-SXM-64GB, partition `boost_usr_prod` |

To recreate the exact Python environment:

```bash
cd /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
uv sync
source .venv/bin/activate
```
