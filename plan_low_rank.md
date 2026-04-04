# Plan: Spectron Low-Rank GPT Training

**Reference**: Janson, Oyallon, Belilovsky. *Stabilizing Native Low-Rank LLM Pretraining*. arXiv:2602.12429, 2025.

---

## Problem Statement

Training a Transformer from scratch with fully-factorized weights `W = AB^T`
(A ∈ R^(m×r), B ∈ R^(n×r), r < min(m,n)) is unstable. The cause:

**Scaling invariance**: for any λ ≠ 0, `(λA)(λ^{-1}B)^T = AB^T`. Independent updates
to A and B allow λ → ∞, driving the spectral norm `||W||_2 → ∞`, causing activation
explosion and training divergence.

Without correction, the spectral norm of `ΔW = ΔA B^T + A ΔB^T + ΔA ΔB^T` grows
10–30× larger in low-rank vs dense training (paper Fig. 2).

---

## Spectron Solution

Two mechanisms combined:

1. **Gradient Orthogonalization** (Newton–Schulz): orthogonalize the momentum buffer
   for each factor — bounds individual factor updates within a local radius ρ.

2. **Spectral Renormalization**: set ρ = η / (σ_A + σ_B + 1) adaptively, where σ_A, σ_B
   are the spectral norms of the current A and B. This ensures the composite update
   satisfies `||ΔW||_2 ≤ ρ(||A||_2 + ||B||_2 + 1) ≤ η` (via submultiplicativity +
   triangle inequality, Eq. 15–16 of paper).

---

## Core Algorithms

### Algorithm 1 — Spectron (per weight pair A, B)

```
Inputs: A ∈ R^(m×r), B ∈ R^(n×r), step size η, momentum β
Init:   M_A = 0, M_B = 0, u_A ∈ R^m (normalized), u_B ∈ R^n (normalized)

Per step t:
  1. G_A ← ∇_A L,  G_B ← ∇_B L
  2. M_A ← β M_A + (1-β) G_A      ← EMA momentum
     M_B ← β M_B + (1-β) G_B
  3. O_A ← NewtonSchulz(M_A)       ← orthogonalized momentum (Algorithm 2)
     O_B ← NewtonSchulz(M_B)
  4. σ_A, u_A ← PowerIter(A, u_A) ← spectral norm estimate (Algorithm 3)
     σ_B, u_B ← PowerIter(B, u_B)
  5. ρ  ← η / (σ_A + σ_B + 1)     ← adaptive scaling
  6. A  ← A - ρ · O_A
     B  ← B - ρ · O_B
```

**Critical note on momentum**: Spectron uses EMA `M ← β M + (1-β) G`, NOT standard
momentum `M ← β M + G`. The (1-β) scaling bounds the buffer magnitude and is required
for the spectral stability proof.

### Algorithm 2 — Newton–Schulz Orthogonalization

Computes the approximate sign/zero-power map on G (nearest orthogonal matrix to G):

```
a, b, c = (3.4445, -4.7750, 2.0315)   ← degree-5 Chebyshev polynomial coefficients
X ← G / (||G||_F + ε)                 ← normalize
if rows > cols: X ← X^T               ← work with narrower dimension
for k_ns iterations:
    A ← X X^T              ← (n×n); paper uses A = X^T X (r×r) — both are equivalent
    B ← b A + c A²
    X ← a X + B @ X       ← left-multiply (code form); paper uses right X @ B
if transposed: X ← X^T
return X
```

> **Left vs right multiply**: The paper uses `A = X^T X` (r×r) and right-multiply `X = a X + X B`.
> The code uses `A = X X^T` (n×n) and left-multiply `X = a X + B @ X`. Both forms are
> mathematically equivalent polynomial approximations of the sign map on singular values.
> The code form is used by `zeropower_via_newtonschulz5` in `train_gpt.py` and reused here.
> The u_A, u_B seed vectors for power iteration are initialized as random normalized vectors
> in optimizer state and warm-started across training steps.

This is **identical** to the `zeropower_via_newtonschulz5` function already in `train_gpt.py`
(used by Muon). Reuse it directly.

Default: k_ns = 5 iterations.

### Algorithm 3 — Power Iteration (Spectral Norm Estimation)

```
u ← u / ||u||
v ← W^T u / ||W^T u||
for k_power iterations:
    u ← W v / ||W v||
    v ← W^T u / ||W^T u||
σ ← u^T (W v)           ← Rayleigh quotient
return σ, u
```

u is maintained in optimizer state across steps (warm-started = faster convergence).
Default: k_power = 1 (cheap: 2 matmuls per factor per step).

---

## Architecture Changes

### LowRankLinear (replaces CastedLinear for all non-embedding layers)

```python
class LowRankLinear(nn.Module):
    # W = A @ B^T,  A: (out, rank),  B: (in, rank)
    # Forward: x → x @ B @ A^T  (two cheap matmuls, avoids materializing W)
    # Float32 params, cast to x.dtype at compute time
    # Spectral init: nn.init.orthogonal_ on both A and B → ||A||_2 = ||B||_2 = 1
    # Zero-init proj layers: orthogonal_(A), zeros_(B) → W = 0 at init
```

**Rank selection**: `rank = max(1, int(rank_ratio * min(out, in)))`.
Default `rank_ratio = 0.25`. Paper tests: 0.4n (best), 0.25n (good), 0.125n (too low).

**Apply to**: all attention projections (Q, K, V, O) and MLP projections (up, down).
**Do NOT apply to**: token embedding, LM head, scalar/vector parameters (RMSNorm, q_gain, etc.).

### Rank defaults for the baseline model (model_dim=512)
| Layer         | Shape (out × in) | rank_ratio=0.25 | Param reduction |
|---------------|-----------------|-----------------|-----------------|
| Q proj        | 512 × 512       | r=128           | 50%             |
| K/V proj      | 256 × 512       | r=64            | 67%             |
| O proj        | 512 × 512       | r=128           | 50%             |
| MLP up        | 1024 × 512      | r=128           | 75%             |
| MLP down      | 512 × 1024      | r=128           | 75%             |

---

## Optimizer Design

| Parameter group       | Optimizer      | LR env var      |
|-----------------------|---------------|-----------------|
| LowRankLinear (A, B)  | **Spectron**   | `MATRIX_LR`     |
| tok_emb (tied)        | Adam           | `TIED_EMBED_LR` |
| tok_emb (untied)      | Adam           | `EMBED_LR`      |
| lm_head (untied)      | Adam           | `HEAD_LR`       |
| scalars/vectors       | Adam           | `SCALAR_LR`     |

Muon is removed. Spectron inherits the same momentum warmup schedule (0.85 → 0.95).

The Spectron optimizer takes a list of `(A, B)` parameter **pairs** (not individual params)
to enforce the joint spectral constraint `ρ = η / (σ_A + σ_B + 1)`.

---

## Hyperparameter Recommendations

| Parameter           | Default   | Notes                                  |
|---------------------|-----------|----------------------------------------|
| `RANK_RATIO`        | 0.25      | Paper: 0.4 is better, 0.25 is a safe trade-off |
| `MATRIX_LR` (η)     | 0.04      | Paper stable range: 1e-3 to 1e-1      |
| `MUON_MOMENTUM` (β) | 0.95      | EMA decay for momentum buffer          |
| `SPECTRON_NS_STEPS` | 5         | Newton-Schulz iterations               |
| `SPECTRON_POWER_ITER_STEPS` | 1 | Power iterations for σ estimation   |

---

## Initialization

**Spectral init** (Khodak et al. 2021): initialize A and B with `nn.init.orthogonal_`.
This gives `||A||_2 = ||B||_2 = 1` and `||W||_2 = ||AB^T||_2 ≤ 1` at initialization.
This prevents rank collapse and provides stable initial spectral norms for power iteration.

**Zero-init proj layers** (same convention as dense model): A = orthogonal, B = zeros,
so W = AB^T = 0 at init (residual-friendly initialization).

---

## Quantization Notes

The state dict contains `.A` and `.B` tensors (both 2D float) instead of `.weight`.
The existing per-row int8 quantization pipeline handles these automatically.
The compressed artifact stores factorized weights — no need to reconstruct W = AB^T at export.
This is the main compression benefit: storing r(m+n) instead of mn elements.

At `rank_ratio=0.25` for square matrices: 50% parameter reduction before quantization.
After int8 + zlib: expected ~4× total compression, fitting comfortably within the 16MB limit.

---

## Expected Results

From paper (FineWeb, 100B-token training, 454M factorized vs 780M dense):
- Factorized (454M) matches Dense (780M) at equal FLOPs
- Naive low-rank (AdamW): perplexity 14.57 (diverges at high lr)
- Self-guided training: perplexity 13.70 (+25% FLOP overhead)
- **Spectron**: perplexity **12.11** (<1% overhead)

Scaling laws nearly match Chinchilla: `N_opt ∝ C^0.479` vs Chinchilla's `C^0.49`.

---

## Implementation File

`train_gpt_low_rank.py` — based on `train_gpt.py` with:
- `LowRankLinear` replacing `CastedLinear` in attention/MLP
- `Spectron` optimizer replacing `Muon`
- `power_iteration` helper function
- `rank_ratio`, `spectron_ns_steps`, `spectron_power_iter_steps` hyperparameters
- Unchanged: data loading, BPB eval, quantization, distributed training, logging

Run command (same interface as train_gpt.py):
```bash
# Smoke test
RUN_ID=lr_test ITERATIONS=200 RANK_RATIO=0.25 torchrun --standalone --nproc_per_node=1 train_gpt_low_rank.py

# Full run on 8×H100
torchrun --standalone --nproc_per_node=8 train_gpt_low_rank.py
```
