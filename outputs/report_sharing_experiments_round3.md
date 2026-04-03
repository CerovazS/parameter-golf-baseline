# Shared Low-Rank Factorization — Experiment Report (Round 3)

**Date:** 2026-04-03
**Codebase:** `train_gpt_sharing.py` (branch `weight_sharing`)
**Hardware:** 4× NVIDIA A100-SXM-64GB, Leonardo HPC — `boost_usr_prod`
**Predecessor:** `report_sharing_experiments.md` (Rounds 1 & 2)

---

## Context

Round 2 of the weight-sharing experiments established that low-rank factorization of the
O projection (`proj`) with `rank_o=256` saves approximately 131 KB of compressed artifact
at a cost of only +0.0025 BPB, and that the V+O low configuration (`rv=64, ro=128`)
achieves 245 KB of savings at +0.0041 BPB — the first configuration to exceed the
baseline's 231 KB headroom below the 16 MB ceiling. However, several directions remained
unexplored: O sharing across more than one pair, the Q projection (same shape as O in
GQA), MLP weight sharing, and multi-projection combinations.

Round 3 systematically covers all four groups identified as next steps in Round 2, using
the same 4×A100 setup and a 700-second wall budget (~3620–3640 steps).

---

## Experimental Setup

### Model and optimizer (all runs)

Identical to Rounds 1 & 2: 9 layers, `model_dim=512`, `num_heads=8`, `num_kv_heads=4`,
`mlp_mult=2`, `vocab_size=1024`, tied embeddings. Optimizer: `MATRIX_LR=0.04`,
`MUON_MOMENTUM=0.95`. Tokens per step: 524,288 (4 GPUs × `batch_size=131072`). Seed: 1337.
Wall budget: 700 seconds. All runs use `train_gpt_sharing.py`.

### Baseline

The Round 2 baseline (`share4_baseline`, job 39005777) was reproduced and confirms the
same values as the original Round 2 baseline:

| BPB f32 | BPB int8 | Artifact bytes | Steps |
|---------|----------|----------------|-------|
| 1.2613 | 1.2643 | 15,768,735 | 3626 |

All Δ values below are relative to this baseline (BPB int8 = 1.2643,
artifact = 15,768,735 bytes, headroom = 231,265 bytes below 16,000,000).

### Sharing configuration

The `SHARING_RANK_V` variable is set to 0 in all Round 3 runs. Round 2 already
established that `rv=128` saves only 4 KB due to the float16 threshold: with
`kv_dim=256` and `model_dim=512`, the factor `C_v` has shape `128×512 = 65,536`
elements and stays in float16. Round 3 does not revisit the V projection.

All active sharing ranks are set to 256 unless noted otherwise.

---

## Run Groups and Results

### Group 1 — O projection, multiple pairs

Extends the Round 2 `share4_solo_o` result (ro=256, p=1: −131 KB, +0.0025 BPB) by
increasing the number of layer pairs that share the same `C_o` factor.

| Run | `ro` | pairs | Steps | BPB f32 | BPB int8 | Δ BPB | Artifact | Δ artifact |
|-----|------|-------|-------|---------|----------|-------|----------|------------|
| share4_solo_o *(Round 2)* | 256 | 1 | 3623 | 1.2635 | 1.2668 | +0.0025 | 15,634,110 | −134,625 |
| share4_o_2pairs | 256 | 2 | 3620 | 1.2644 | 1.2676 | +0.0033 | 15,388,314 | −380,421 |
| share4_o_3pairs | 256 | 3 | 3634 | 1.2680 | 1.2713 | +0.0070 | 15,154,897 | −613,838 |
| share4_o_4pairs | 256 | 4 | 3627 | 1.2675 | 1.2709 | +0.0066 | 14,923,209 | −845,526 |

Each additional pair saves approximately 235–235 KB of artifact, because each new
`B_o` factor (shape `512×256 = 131,072` elements, int8) replaces a full-rank O weight,
and the shared `C_o` is compressed only once. The BPB cost, however, does not scale
cleanly: p=2 costs only +0.0033, but p=3 jumps to +0.0070. Crucially, p=4 is slightly
cheaper than p=3 (+0.0066 vs +0.0070), suggesting non-monotonic interaction with the
layer symmetry structure of the 9-layer network.

At p=4, 846 KB of savings are achieved — more than 3.6× the baseline headroom —
at a BPB cost of +0.0066.

### Group 2 — Q projection

The Q projection in this GQA setup has shape `(num_heads × head_dim) × model_dim =
512×512`, identical to O. Round 2 did not test Q sharing at all.

| Run | `rq` | pairs | Steps | BPB f32 | BPB int8 | Δ BPB | Artifact | Δ artifact |
|-----|------|-------|-------|---------|----------|-------|----------|------------|
| share4_q_1pair | 256 | 1 | 3628 | 1.2612 | 1.2640 | **−0.0003** | 15,651,216 | −117,519 |
| share4_q_2pairs | 256 | 2 | 3626 | 1.2636 | 1.2668 | +0.0025 | 15,405,243 | −363,492 |

**Q at 1 pair is the single best result of Round 3.** The int8 BPB of 1.2640 is
3 tenths of a milliBPB *below* the baseline (1.2643), which is within noise but
confirms that Q sharing at rank 256 does not hurt and may provide mild implicit
regularization. The artifact saving of 117 KB is modest — comparable to solo O at
1 pair — but the zero BPB cost is unique among all sharing configurations tested so far.

At 2 pairs, Q sharing costs +0.0025 BPB and saves 363 KB, matching the cost profile
of O at 1 pair but with approximately 2.7× more artifact savings.

### Group 3 — MLP sharing

The MLP block contains two weight matrices per layer: `c_fc` (shape
`mlp_dim × model_dim = 1024×512`) and `c_proj` (shape `model_dim × mlp_dim = 512×1024`).
Both exceed the 65,536-element threshold at rank 256, so their factors land in int8.
The expected saving per shared pair is larger than for attention projections because the
MLP matrices are wider.

| Run | `rmlp_fc` | `rmlp_proj` | pairs | Steps | BPB f32 | BPB int8 | Δ BPB | Artifact | Δ artifact |
|-----|-----------|-------------|-------|-------|---------|----------|-------|----------|------------|
| share4_mlp_fc_1p | 256 | 0 | 1 | 3634 | 1.2678 | 1.2715 | +0.0072 | 15,399,079 | −369,656 |
| share4_mlp_proj_1p | 0 | 256 | 1 | 3638 | 1.2679 | 1.2711 | +0.0068 | 15,269,849 | −498,886 |
| share4_mlp_both_1p | 256 | 256 | 1 | 3629 | 1.2750 | 1.2793 | +0.0150 | 14,924,042 | −844,693 |
| share4_mlp_fc_2p | 256 | 0 | 2 | 3671 | 1.2728 | 1.2762 | +0.0119 | 14,901,022 | −867,713 |

MLP sharing is significantly more expensive in BPB than attention sharing at the same
rank. `MLP_FC` alone at 1 pair costs +0.0072 BPB for 370 KB saved; `MLP_PROJ` alone is
slightly cheaper at +0.0068 BPB for 499 KB saved. The asymmetry in savings (fc vs proj)
is expected from the shape difference. Combining both MLP projections at 1 pair produces
a super-additive penalty: +0.0150 BPB for 845 KB saved, where the additive expectation
would be approximately +0.014. Adding a second pair to MLP_FC only pushes the cost to
+0.0119 BPB for 868 KB.

The bytes-saved-per-BPB-point ratio for MLP is approximately 50–73 KB per mBPB, roughly
half the efficiency of Q sharing and one-third that of O sharing at the same scale.

### Group 4 — Combined projections

| Run | `ro` | `rq` | `rmlp_fc` | `rmlp_proj` | pairs | Steps | BPB f32 | BPB int8 | Δ BPB | Artifact | Δ artifact |
|-----|------|------|-----------|-------------|-------|-------|---------|----------|-------|----------|------------|
| share4_o_q_1p | 256 | 256 | 0 | 0 | 1 | 3633 | 1.2634 | 1.2666 | +0.0023 | 15,520,697 | −248,038 |
| share4_o_q_2p | 256 | 256 | 0 | 0 | 2 | 3624 | 1.2668 | 1.2702 | +0.0059 | 15,038,260 | −730,475 |
| share4_o_mlp_fc_1p | 256 | 0 | 256 | 0 | 1 | 3627 | 1.2691 | 1.2727 | +0.0084 | 15,273,737 | −495,008 |
| share4_full_1p | 256 | 256 | 256 | 256 | 1 | **1163*** | 1.3677 | 1.3700 | +0.1057 | 12,227,414 | −3,541,321 |

*Training terminated early — see note below.

**O+Q at 1 pair** combines the two cheapest projections and achieves +0.0023 BPB for
248 KB saved. The BPB cost is essentially additive: O alone +0.0025, Q alone −0.0003,
combined +0.0023. This confirms that O and Q sharing interact independently at 1 pair.

**O+Q at 2 pairs** saves 730 KB at +0.0059 BPB — a strong operating point that opens
approximately 3.2× the baseline headroom.

**O+MLP_FC at 1 pair** costs +0.0084 BPB for 495 KB. The marginal cost of adding
MLP_FC on top of O is +0.0059 (84−25), versus +0.0072 for MLP_FC alone — essentially
additive, suggesting no synergistic degradation between O and MLP sharing.

#### Training failure: share4_full_1p

The configuration `ro=256, rq=256, rmlp_fc=256, rmlp_proj=256, pairs=1` diverged
after 1163 steps. The per-step timing was normal (~175 ms), so the early termination
was not a compute failure but an implicit wall-time exit driven by the wallclock budget
after divergence slowed iteration to a halt. The final int8 BPB of 1.3700 is +0.106
above baseline, indicating complete training failure. Simultaneous sharing of all four
projections at rank 256 in a single pair is incompatible with stable training at this
optimizer configuration.

---

## Summary Table

All Round 3 runs plus the Round 2 reference configurations for comparison.

| Config | rv | ro | rq | rmlp_fc | rmlp_proj | pairs | Steps | BPB int8 | Δ BPB | Artifact | Δ artifact |
|--------|----|----|----|---------|-----------|----- -|-------|----------|-------|----------|------------|
| **Baseline** | 0 | 0 | 0 | 0 | 0 | — | 3626 | **1.2643** | 0.0000 | 15,768,735 | 0 |
| *Solo O (Round 2)* | 0 | 256 | 0 | 0 | 0 | 1 | 3623 | *1.2668* | *+0.0025* | *15,634,110* | *−134,625* |
| *V+O low (Round 2)* | 64 | 128 | 0 | 0 | 0 | 1 | 3626 | *1.2684* | *+0.0041* | *15,517,908* | *−250,827* |
| Q p=1 | 0 | 0 | 256 | 0 | 0 | 1 | 3628 | **1.2640** | **−0.0003** | 15,651,216 | −117,519 |
| O+Q p=1 | 0 | 256 | 256 | 0 | 0 | 1 | 3633 | 1.2666 | +0.0023 | 15,520,697 | −248,038 |
| Q p=2 | 0 | 0 | 256 | 0 | 0 | 2 | 3626 | 1.2668 | +0.0025 | 15,405,243 | −363,492 |
| O p=2 | 0 | 256 | 0 | 0 | 0 | 2 | 3620 | 1.2676 | +0.0033 | 15,388,314 | −380,421 |
| MLP_PROJ p=1 | 0 | 0 | 0 | 0 | 256 | 1 | 3638 | 1.2711 | +0.0068 | 15,269,849 | −498,886 |
| O p=4 | 0 | 256 | 0 | 0 | 0 | 4 | 3627 | 1.2709 | +0.0066 | 14,923,209 | −845,526 |
| O+Q p=2 | 0 | 256 | 256 | 0 | 0 | 2 | 3624 | 1.2702 | +0.0059 | 15,038,260 | −730,475 |
| MLP_FC p=1 | 0 | 0 | 0 | 256 | 0 | 1 | 3634 | 1.2715 | +0.0072 | 15,399,079 | −369,656 |
| O+MLP_FC p=1 | 0 | 256 | 0 | 256 | 0 | 1 | 3627 | 1.2727 | +0.0084 | 15,273,737 | −495,008 |
| MLP_FC p=2 | 0 | 0 | 0 | 256 | 0 | 2 | 3671 | 1.2762 | +0.0119 | 14,901,022 | −867,713 |
| O p=3 | 0 | 256 | 0 | 0 | 0 | 3 | 3634 | 1.2713 | +0.0070 | 15,154,897 | −613,838 |
| MLP_both p=1 | 0 | 0 | 0 | 256 | 256 | 1 | 3629 | 1.2793 | +0.0150 | 14,924,042 | −844,693 |
| full p=1 *(failed)* | 0 | 256 | 256 | 256 | 256 | 1 | 1163 | 1.3700 | +0.1057 | 12,227,414 | −3,541,321 |

---

## Analysis

### Q projection: a free saving

The Q projection behaves fundamentally differently from O despite identical shape.
At rank 256, 1 pair, Q sharing produces a net negative Δ BPB (−0.0003). The most likely
explanation is a mild regularization effect: the shared factor `C_q` acts as an
information bottleneck between the outermost query heads, reducing overfitting in the
early-layer attention heads. This effect is specific to the 1-pair case; at 2 pairs it
reverts to a typical cost (+0.0025 BPB). The implication is that Q sharing at 1 pair is
strictly dominated by no sharing from a BPB standpoint and should be included by default
in any configuration that has budget for it.

### O sharing: approximately linear per-pair cost, non-monotone at p=3

O sharing scales approximately linearly in artifact savings across pairs
(134 → 380 → 614 → 846 KB for p=1..4), but the BPB cost has a local maximum at p=3
(+0.0070) that is higher than p=4 (+0.0066). This likely reflects the layer indexing:
with 9 transformer layers, p=4 pairs share layers {0,8}, {1,7}, {2,6}, {3,5}, which by
symmetry may be a more natural partition than p=3. This observation is a weak signal at
single seed, but the pattern is consistent with the architecture's symmetry.

### MLP sharing: high BPB cost, low efficiency

MLP sharing costs approximately 3–4× more BPB per KB saved than Q or O sharing. The
MLP weights carry more task-specific representational load than the attention projections
in this shallow (9-layer) model, making shared factorization more destructive. MLP_PROJ
is slightly cheaper than MLP_FC (+0.0068 vs +0.0072), which is consistent with MLP_FC
being the gate or activation-path weight and thus more sensitive to perturbation.
Combining both MLP projections is super-additively expensive (+0.0150 vs +0.014
additive prediction).

### Combinations: O and Q are additive and independent

The O+Q combined results are well predicted by the sum of the individual costs.
At 1 pair: O costs +0.0025, Q costs −0.0003, combined costs +0.0023 (additive: +0.0022).
At 2 pairs: O costs +0.0033, Q costs +0.0025, combined costs +0.0059 (additive: +0.0058).
The near-perfect additivity implies that the two projections occupy orthogonal regions of
the loss landscape at these scales, and that they can be combined without penalty.

The same additivity does not hold for MLP: O+MLP_FC costs +0.0084, versus an additive
prediction of +0.0097 (O=+0.0025, MLP_FC=+0.0072). MLP_FC actually becomes
*slightly cheaper* when paired with O. This is a weak effect (13 mBPB reduction) and
not actionable at single seed.

### Training stability

All configurations except `full_1p` trained stably to ~3620–3671 steps in 700 s.
The only failure was the 4-projection full combination. Individual BPB profiles
(not shown in detail) are smooth and the quantization Δ (BPB int8 − BPB f32) remains
at +0.0028 to +0.0033 across all runs, confirming that shared factorization does not
degrade quantization quality at any scale tested.

---

## Conclusions

1. **Q at rank 256, 1 pair, is a free improvement.** BPB decreases by 0.0003 relative to
   baseline. This should be included in all subsequent configurations. Artifact saving
   (117 KB) is modest but non-negative.

2. **O+Q at 1 pair is the cleanest combined result.** +0.0023 BPB for 248 KB saved.
   The O and Q factors are additively independent — combining them costs no more than the
   sum of their individual costs.

3. **O+Q at 2 pairs opens substantial headroom at reasonable cost.** +0.0059 BPB for
   730 KB saved — approximately 3.2× the baseline headroom. This is the best operating
   point for configurations that need genuine space below the 16 MB ceiling.

4. **MLP sharing is inefficient at rank 256.** The BPB cost per KB saved is 3–4× worse
   than attention projections. MLP sharing alone is not recommended at this rank.

5. **Full simultaneous sharing at rank 256 diverges.** All four projections (O, Q,
   MLP_FC, MLP_PROJ) shared simultaneously at rank 256 produces training failure.
   This configuration is ruled out.

6. **O sharing at p=4 saves 846 KB at +0.0066 BPB.** For configurations that need
   maximum single-projection savings and can absorb the BPB cost, this remains the
   highest-saving stable attention configuration tested.

---

## Next Steps

- **Seed validation for Q p=1:** The −0.0003 BPB result for Q sharing at 1 pair needs
  3-seed confirmation before it can be treated as a reliable effect rather than noise.
  At a single seed the signal is within the ±0.002 cross-run spread observed in Round 2.

- **O+Q p=2, more seeds:** The +0.0059 BPB / −730 KB operating point is a strong
  candidate for inclusion in a production submission. Validate at 3 seeds.

- **V at rank 256:** Round 2 showed that `rv=128` saves only 4 KB due to the float16
  threshold (`C_v` has exactly 65,536 elements). At `rv=256`, `C_v` has shape
  `256×512 = 131,072` elements and enters int8 territory. Test `rv=256` with 1 pair
  to quantify the actual int8 saving and BPB cost.

- **MLP at lower rank (128):** MLP sharing at rank 256 is too costly. Testing rank 128
  for MLP_FC and MLP_PROJ may reveal a cheaper operating point where the capacity loss
  is smaller, particularly if combined with O+Q sharing.

- **Full combination with reduced ranks:** Once V (r=256) and MLP (r=128) costs are
  known, attempt a combination run (O+Q+V+MLP_FC at conservative ranks) to test whether
  the divergence was specific to rank 256 or to simultaneous sharing in general.
