# Program: GNS Muon v2 Benchmarks

## Objective

Validate the Gram-Newton-Schulz Muon optimizer integration with:
1. torch.compile patches (already applied to venv)
2. LR adjustment fix (already in train_gpt_exp.py)

Then iterate on hyperparameters to maximize step throughput × convergence quality.

## Phase 1: Baseline + GNS v2 comparison (12-min runs)

Run pairs of baseline vs GNS at each dimension:

| # | Script | Dim | RUN_ID | Env Overrides |
|---|--------|-----|--------|---------------|
| 1 | train_gpt.py | 512 | baseline_v2_512 | (defaults) |
| 2 | train_gpt_exp.py | 512 | gns_v2_512 | (defaults) |
| 3 | train_gpt.py | 768 | baseline_v2_768 | MODEL_DIM=768 |
| 4 | train_gpt_exp.py | 768 | gns_v2_768 | MODEL_DIM=768 |
| 5 | train_gpt.py | 1024 | baseline_v2_1024 | MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 |
| 6 | train_gpt_exp.py | 1024 | gns_v2_1024 | MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 |

All: MAX_WALLCLOCK_SECONDS=720

## Phase 2: Hyperparameter tuning for GNS Muon

After Phase 1 confirms speed+quality parity, explore:
- matrix_lr sweep (GNS may benefit from different LR)
- momentum tuning
- NS coefficients (YOU vs POLAR_EXPRESS)
- Warmdown schedule optimization

## Baselines to beat
- dim=512: 366ms/step, val_bpb=1.2973
- dim=768: 687ms/step, val_bpb=1.3035
- dim=1024: 1280ms/step, val_bpb=1.3648
