# plan.md — Gram-Newton-Schulz Integration into train_gpt_exp.py

## Objective

Replace the standard Newton-Schulz orthogonalization step in Muon with Gram-Newton-Schulz
from https://github.com/Dao-AILab/gram-newton-schulz, targeting the 2x RTX 5090 (Blackwell, sm_100).

---

## Background: Algorithm Differences

### Standard Newton-Schulz (current, `zeropower_via_newtonschulz5`)
- Fixed coefficients: `a=3.4445, b=-4.7750, c=2.0315` (same for all iterations)
- Iterates on full rectangular matrix X ∈ R^(n×m)
- Per iteration: `A = X Xᵀ`, `B = bA + cA²`, `X ← aX + BX`  — one rectangular GEMM per iter
- Cast to **bfloat16**
- FLOPs: ~65n³ at T=5, α=4

### Gram-Newton-Schulz (new)
- **Time-varying** coefficients (Polar Express, 5 distinct `[a,b,c]` triples)
- Iterates on Gram matrix R = XXᵀ ∈ R^(n×n) (symmetric, small) + accumulates Q
- One rectangular GEMM at init (`XXᵀ`) + one at end (`QX`), all intermediate work is n×n
- Cast to **float16** (higher mantissa precision than bf16 → fewer spurious negative eigenvalues)
- **Restart after iteration 2** to prevent eigenvalue divergence in fp16
- Fallback to standard 3-GEMM loop for square matrices (no Gram benefit)
- FLOPs: ~38n³ at T=5, α=4 → **~42% fewer FLOPs** for typical weight shapes
- RTX 5090 (sm_100 Blackwell): custom symmetric GEMM kernels available if package installed

---

## Coefficient Presets

### POLAR_EXPRESS_COEFFICIENTS (default)
From arxiv 2505.16932 with safety_factor=1.05:
```
sf = 1.05
[a/sf, b/sf³, c/sf⁵] for each of 5 iterations:
iter 0: (7.892583, -20.477034, 13.093825)
iter 1: (3.911485, -2.563779, 0.412063)
iter 2: (3.760658, -2.529307, 0.417532)
iter 3: (3.160400, -2.163620, 0.386052)
iter 4: (2.191097, -1.451968, 0.317032)
```

### YOU_COEFFICIENTS (alternative)
```
iter 0: (4.0848, -6.8946, 2.9270)
iter 1: (3.9505, -6.3029, 2.6377)
iter 2: (3.7418, -5.5913, 2.3037)
iter 3: (2.8769, -3.1427, 1.2046)
iter 4: (2.8366, -3.0525, 1.2012)
```

---

## Files Modified

| File | Change |
|---|---|
| `train_gpt_exp.py` | New file: copy of `train_gpt.py` with Gram-NS substitution |

---

## Precise Diff Plan

### 1. Add coefficient constants after imports (before line 93)

```python
# Gram-Newton-Schulz Polar Express coefficients (arxiv 2505.16932, safety_factor=1.05)
_SF = 1.05
GRAM_NS_COEFFICIENTS = [
    ( 8.28721201814563/_SF, -23.595886519098837/_SF**3,  17.300387312530933/_SF**5),
    ( 4.107059111542203/_SF,  -2.9478499167379106/_SF**3,  0.5448431082926601/_SF**5),
    ( 3.9486908534822946/_SF,  -2.908902115962949/_SF**3,  0.5518191394370137/_SF**5),
    ( 3.3184196573706015/_SF,  -2.488488024314874/_SF**3,   0.51004894012372/_SF**5),
    ( 2.300652019954817/_SF,  -1.6689039845747493/_SF**3,  0.4188073119525673/_SF**5),
]
```

### 2. Replace `zeropower_via_newtonschulz5` (lines 100–113) with `gram_newton_schulz`

```python
def gram_newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Gram-Newton-Schulz orthogonalization (replaces zeropower_via_newtonschulz5).

    Instead of iterating on the full rectangular matrix X ∈ R^(n×m), iterates on
    the Gram matrix R = XXᵀ ∈ R^(n×n) and separately accumulates the orthogonal
    factor Q. Only two rectangular GEMMs total (init + finalize) vs one per
    iteration in standard NS.

    Key changes vs standard NS:
    - Time-varying Polar Express coefficients (5 distinct [a,b,c] triples)
    - float16 (not bfloat16): better mantissa precision reduces spurious negatives
    - Restart after iteration 2 (0-indexed): prevents eigenvalue divergence
    - Falls back to standard 3-GEMM loop for square matrices

    FLOPs: ~38n³ vs ~65n³ for standard NS at T=5, aspect ratio α=4.
    Hardware: RTX 5090 (sm_100 Blackwell) — Gram symmetric GEMMs map well to
    Blackwell tensor memory; custom kernels active if gram_newton_schulz package
    is installed.
    """
    coeff_list = GRAM_NS_COEFFICIENTS[:steps]

    X = G.to(torch.float16)
    X = X / (X.norm() + eps)

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    n, m = X.shape

    if n == m:
        # Square matrix: standard 3-GEMM loop (Gram offers no benefit)
        for a, b, c in coeff_list:
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    else:
        # Non-square: Gram iteration on R = XXᵀ ∈ R^(n×n)
        R = X @ X.T
        Q = torch.eye(n, dtype=X.dtype, device=X.device)

        for i, (a, b, c) in enumerate(coeff_list):
            if i == 2:
                # Restart: materialize Q into X, recompute Gram from scratch.
                # Prevents accumulation of fp16 rounding errors across Q updates.
                X = Q @ X
                R = X @ X.T
                Q = torch.eye(n, dtype=X.dtype, device=X.device)

            # Z = bR + cR² (a*I is distributed across Q/R updates to avoid fp16 cancellation)
            R2 = R @ R
            Z = b * R + c * R2

            # Q ← Q(Z + aI) = QZ + aQ
            Q = Q @ Z + a * Q
            # RZ = R(Z + aI) = RZ + aR
            RZ = R @ Z + a * R
            # R ← (Z + aI)RZ = Z@RZ + a*RZ
            R = Z @ RZ + a * RZ

        X = Q @ X  # single final rectangular GEMM

    if transposed:
        X = X.T

    return X
```

### 3. Update call site in `Muon.step()` (line 158)

```python
# Before:
g = zeropower_via_newtonschulz5(g, steps=backend_steps)
# After:
g = gram_newton_schulz(g, steps=backend_steps)
```

### 4. Update `torch.compile` target (lines 737–741)

```python
# Before:
global zeropower_via_newtonschulz5
zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
# After:
global gram_newton_schulz
gram_newton_schulz = torch.compile(gram_newton_schulz)
```

### 5. Optional: try gram-newton-schulz package for kernel-accelerated path

At top of file, after torch imports:
```python
try:
    from gram_newton_schulz import GramNewtonSchulz as _GramNSKernel
    _gram_ns_kernel = _GramNSKernel(ns_use_kernels=True)
    _USE_GRAM_NS_KERNELS = True
except ImportError:
    _USE_GRAM_NS_KERNELS = False
```

And inside `gram_newton_schulz`, before the manual implementation:
```python
if _USE_GRAM_NS_KERNELS:
    return _gram_ns_kernel(G)
```

---

## What Does NOT Change

- All DDP/distributed logic in `Muon.step()` — unchanged
- Scale correction `g *= max(1, g.size(0)/g.size(1))**0.5` — unchanged
- All optimizer instantiation, LR schedules, momentum warmup — unchanged
- Model architecture — unchanged
- Data pipeline — unchanged

---

## Hardware Notes (RTX 5090 / Blackwell sm_100)

- Gram NS Gram matrix iterations are all n×n symmetric GEMMs — map ideally to
  Blackwell 2-CTA collaboration model and tensor memory hierarchy
- Pure PyTorch path: speedup from ~42% fewer FLOPs (38n³ vs 65n³)
- If `gram-newton-schulz` package installable (needs Python 3.12+, CUDA 12.9+):
  additional speedup from custom symmetric GEMM kernels (triangular scheduler,
  transposed epilogue) for n > 256
- Currently: Python 3.11 / CUDA 13.1 → package may install; try with
  `pip install /root/gram-newton-schulz --no-build-isolation`
- `torch.compile` on Blackwell: set `TORCH_COMPILE_DISABLE=1` if compilation hangs

---

## Launch Command

```bash
cd ~/parameter-golf-baseline
source .venv/bin/activate

HYPOTHESIS="gram-newton-schulz polar-express coefficients, fp16, restart@2" \
ITERATIONS=1000 \
WARMDOWN_ITERS=100 \
./tools/launch_run.sh --script train_gpt_exp.py --gpus 2 --local
```
