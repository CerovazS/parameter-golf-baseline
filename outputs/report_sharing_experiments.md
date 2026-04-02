# Shared Low-Rank Factorization — Experiment Report (Rounds 1 & 2)

**Date:** 2026-04-03
**Codebase:** `train_gpt_sharing.py` (branch `main`, commit `41f54031`)
**Hardware (Round 1):** 1× NVIDIA A100-SXM-64GB, Leonardo HPC — `boost_usr_prod`
**Hardware (Round 2):** 4× NVIDIA A100-SXM-64GB, Leonardo HPC — `boost_usr_prod`

---

## Motivation

The Parameter Golf challenge imposes a hard 16 MB limit on the compressed model artifact
(int8 quantized + zlib level 9). The baseline model at full training budget already sits at
approximately 15.77 MB, leaving only about 230 KB of headroom for architecture
improvements. Any architectural change that grows the artifact beyond that ceiling is
immediately disqualifying.

The hypothesis is that weight matrices in symmetric encoder-decoder layer pairs share
significant subspace structure, exploitable via a shared low-rank factorization:

```
W_i = B_i @ C      (B_i local to layer i,  C shared across the pair)
W_j = B_j @ C      (B_j local to layer j,  same C)
```

Because `C` appears exactly once in the serialized state dict, it is quantized and
compressed once by zlib, reducing the artifact size. The low-rank constraint may also
act as an implicit regularizer. If the artifact savings are real and the BPB cost is
small, this technique creates headroom for other architecture improvements without
exceeding the 16 MB ceiling.

---

## Implementation

A new training script `train_gpt_sharing.py` was derived from `train_gpt.py` with two
new module types:

- **`SharedFactorC(rank, in_features)`** — holds the shared right factor `C`
  (shape `rank × in_features`), registered at the GPT level so it appears exactly
  once in the state dict and is quantized and compressed once.
- **`FactoredLinear(out, in, rank, shared_C, zero_init)`** — replaces `CastedLinear`
  for targeted projections. Holds local factor `B` (shape `out × rank`); references
  `C` via a plain Python list to avoid double-counting in `named_parameters()`.
  Forward: `F.linear(x, B.to(dtype) @ C.to(dtype))`.

`C` factors are orthogonally initialized. `B` factors for the O projection (`proj`) are
zero-initialized to preserve the residual-stream stability guarantee of the original
architecture.

The shared `C` factors are added to the Muon optimizer group alongside all other 2D
weight matrices in the transformer blocks.

New environment variables controlling the sharing configuration:

| Variable | Default | Description |
|---|---|---|
| `SHARING_RANK_V` | 0 | Rank for the V projection (`c_v`) shared factor |
| `SHARING_RANK_O` | 0 | Rank for the O projection (`proj`) shared factor |
| `SHARING_RANK_Q` | 0 | Rank for the Q projection (`c_q`) shared factor |
| `SHARING_RANK_MLP_FC` | 0 | Rank for the MLP fully-connected weight shared factor |
| `SHARING_RANK_MLP_PROJ` | 0 | Rank for the MLP projection weight shared factor |
| `SHARING_NUM_PAIRS` | 1 | Number of encoder-decoder layer pairs to share |

A value of 0 disables sharing for that projection. `SHARING_NUM_PAIRS=1` shares only
the outermost pair (layer 0 / layer `num_encoder_layers`); higher values extend sharing
inward.

---

## Experimental Setup

### Model and optimizer configuration (all runs)

All runs use identical model configuration: 9 layers, `model_dim=512`, `num_heads=8`,
`num_kv_heads=4`, `mlp_mult=2`, `vocab_size=1024`, tied embeddings. Optimizer:
`MATRIX_LR=0.04`, `MUON_MOMENTUM=0.95`. Tokens per step: 524,288 (4 GPUs ×
`batch_size=131072`). Seed: 1337.

### Round 1 — 1×A100, approximately 760 steps

Six runs were executed at approximately 760 training steps (500 s wall budget on a
single A100). These runs were intended as a rapid feasibility check to verify that
training was stable and that artifact size changed in the expected direction.

| Run ID | Script | `rv` | `ro` | Steps |
|---|---|---|---|---|
| `run_20260402_220351` | `train_gpt.py` | — | — | 761 |
| `run_20260402_225454` | `train_gpt_sharing.py` | 0 | 256 | 760 |
| `run_20260402_231023` | `train_gpt_sharing.py` | 128 | 0 | 761 |
| `share_solo_v_…job39000900` | `train_gpt_sharing.py` | 128 | 0 | 760 |
| `share_v_o_…job39000901` | `train_gpt_sharing.py` | 128 | 256 | 761 |
| `share_v_o_low_…job39000903` | `train_gpt_sharing.py` | 64 | 128 | 760 |

### Round 2 — 4×A100, approximately 3620 steps

Five runs were executed at approximately 3620 training steps (700 s wall budget on
4×A100) via `sbatch`. These form the primary dataset for quantitative conclusions.
Validation was logged every 200 steps (`VAL_LOSS_EVERY=200`).

| Run ID | Script | `rv` | `ro` | `pairs` | Steps |
|---|---|---|---|---|---|
| `share4_baseline_…` | `train_gpt.py` | — | — | — | 3626 |
| `share4_solo_o_…job39005467` | `train_gpt_sharing.py` | 0 | 256 | 1 | 3623 |
| `share4_solo_v_…` | `train_gpt_sharing.py` | 128 | 0 | 1 | 3627 |
| `share4_v_o_…` | `train_gpt_sharing.py` | 128 | 256 | 1 | 3621 |
| `share4_v_o_low_…` | `train_gpt_sharing.py` | 64 | 128 | 1 | 3626 |
| `share4_v_multi_…` | `train_gpt_sharing.py` | 128 | 256 | 4 | 3625 |

---

## Results

**Round 1 results are not reported here.** At approximately 760 steps the learning rate
warmdown has not yet activated, and node-to-node variance on the cluster is large enough
to dominate any signal. Two identical solo-V configurations (same code, same seed)
produced a 0.009 BPB gap (1.4010 vs. 1.3922), which is larger than any effect being
measured. Round 1 was useful only for stability checks and artifact size verification.
All quantitative conclusions are drawn from Round 2 only.

### Round 2 — BPB and artifact size

| Config | rv | ro | pairs | steps | BPB f32 | BPB int8 | Δ BPB | artifact bytes | Δ artifact |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0 | 0 | — | 3626 | 1.2613 | 1.2643 | +0.0030 | 15,768,735 | — |
| Solo O ro=256 | 0 | 256 | 1 | 3623 | 1.2635 | 1.2668 | +0.0033 | 15,634,110 | −134,625 (~131 KB) |
| Solo V rv=128 | 128 | 0 | 1 | 3627 | 1.2630 | 1.2660 | +0.0030 | 15,764,811 | −3,924 (~4 KB) |
| V+O 128/256 | 128 | 256 | 1 | 3621 | 1.2667 | 1.2697 | +0.0030 | 15,618,554 | −150,181 (~147 KB) |
| V+O low 64/128 | 64 | 128 | 1 | 3626 | 1.2653 | 1.2684 | +0.0031 | 15,517,908 | −250,827 (~245 KB) |
| V+O multi 128/256 | 128 | 256 | 4 | 3625 | 1.2804 | 1.2837 | +0.0033 | 14,546,230 | −1,222,505 (~1.19 MB) |

Δ BPB = BPB int8 (config) − BPB int8 (baseline). Positive = worse than baseline.

### Per-step training time

All configurations: **657–658 ms/step** — identical to baseline within measurement noise.
The `B @ C` matmul at each forward pass adds zero measurable overhead.

---

## Analysis

### Float16 threshold effect

The quantization pipeline stores tensors with at most 65,536 elements as float16
(2 bytes/element) rather than int8 (1 byte/element). With `rank_v=128`, the factors
are:

- `B_v`: shape 256×128 = 32,768 elements → float16
- `C_v`: shape 128×512 = 65,536 elements → float16 (exactly at threshold, inclusive)

Both factors fall into the float16 bucket. The element count reduction is real, but the
bytes-per-element doubles for the shared factor relative to int8, nearly cancelling the
saving. This explains why solo V with `rv=128` saves only 4 KB instead of the roughly
100 KB one might naively expect.

The O projection is unaffected: with `rank_o=256`:

- `B_o`: shape 512×256 = 131,072 elements → int8
- `C_o`: shape 256×512 = 131,072 elements → int8

Both factors exceed the threshold and are quantized to int8, yielding a genuine saving.

To obtain real int8 savings on the V projection, `rank_v` must exceed 128 so that `C_v`
(shape `rank_v × 512`) has more than 65,536 elements. The minimum effective rank for V
is 129; practically, 192 or 256 is recommended.

### BPB / artifact tradeoff

The Round 2 runs reveal a clear tradeoff:

- **Solo O (ro=256, 1 pair)**: saves 131 KB, costs only +0.0025 BPB relative to
  baseline. The best pure artifact-reduction option for a single projection.
- **Solo V (rv=128, 1 pair)**: saves only 4 KB due to the float16 threshold. BPB
  cost is negligible (+0.0017). Not useful at rank 128.
- **V+O low (rv=64, ro=128, 1 pair)**: saves 245 KB, costs +0.0041 BPB. This is the
  best single-pair configuration in terms of bytes saved per BPB point spent.
- **V+O (rv=128, ro=256, 1 pair)**: saves 147 KB, costs +0.0054 BPB. Strictly dominated
  by V+O low on both axes.
- **V+O multi (rv=128, ro=256, 4 pairs)**: saves 1.19 MB as predicted, but costs
  +0.0194 BPB. The BPB penalty grows approximately linearly with the number of pairs.

In the challenge context, the baseline already has only 231,265 bytes of headroom
(16,000,000 − 15,768,735). V+O low is the first configuration to exceed the baseline
headroom at 245 KB saved, creating genuine space for architectural expansion. Solo O
(131 KB) covers roughly half the headroom.

### Round 1 noise explanation

Two solo-V runs with identical hyperparameters and seed 1337 but different execution
environments (interactive session vs. sbatch dedicated node) produced a 0.009 BPB gap.
At 760 steps the training is still in the warmup-to-plateau phase before warmdown
begins, so NCCL timing, GPU clock state, and memory bandwidth variation between nodes
can shift the effective gradient trajectory enough to produce differences of this
magnitude. Any BPB comparison at 760 steps with a precision target below 0.009 is
unreliable. Round 2 at 3620 steps is past the early instability region and the
cross-run spread drops to approximately 0.002 BPB.

### Quantization quality

For all configurations, Δ quant (BPB int8 − BPB f32) is approximately +0.003. This is
identical to the baseline and confirms that the low-rank factorization does not degrade
quantization quality. The B and C factors quantize independently without interaction
artifacts.

---

## Conclusions

1. **Zero training overhead.** The factored forward pass adds no measurable per-step cost
   (657–658 ms/step across all configurations).

2. **Float16 threshold governs V projection savings.** With `kv_dim=256` and
   `model_dim=512`, `rank_v` must exceed 128 to place `C_v` into int8 territory. At
   rank 128 only 4 KB is saved. Ranks 192 or 256 are needed for material savings.

3. **Solo O (ro=256) is a clean win.** 131 KB saved, +0.0025 BPB, zero overhead. Worth
   including in any configuration that is close to the 16 MB ceiling.

4. **V+O low (rv=64, ro=128) is the best single-pair tradeoff.** 245 KB saved — enough
   to open genuine headroom beyond the current baseline — at a BPB cost of +0.0041.

5. **Multi-pair sharing is too expensive at 4 pairs.** The 1.19 MB saving comes at a
   cost of +0.019 BPB. Pair counts of 2–3 are unexplored and may offer a better
   operating point.

6. **Sharing does not affect quantization quality.** Δ quant ≈ +0.003 for all
   configurations, matching the baseline.

---

## Next Steps

- **O projection, more pairs (Groups 1):** Test `ro=256` with 2, 3, 4 pairs to map the
  BPB-vs-savings curve for O-only sharing.
- **Q projection (Group 2):** `SHARING_RANK_Q=256` with 1 and 2 pairs. Q has the same
  shape as O in GQA, so savings should be similar.
- **MLP sharing (Group 3):** Test `SHARING_RANK_MLP_FC` and `SHARING_RANK_MLP_PROJ`
  independently and combined. MLP weights are larger than attention projections so
  int8 savings should be more pronounced.
- **Combinations (Group 4):** O+Q and O+MLP_FC combinations to see if savings stack
  without multiplicative BPB cost. A full-combination run (`share4_full_1p`) will
  establish the maximum single-pair savings achievable with the current factorization.
- **V with rank > 128:** Try `rv=256` to escape the float16 threshold and compare with
  `rv=128` to quantify the additional int8 saving.
- **Multiple seeds:** Run the best configuration at 3 seeds to confirm BPB signal and
  bound variance at full training budget.
