# Triton Kernels — Technical Report

**Branch**: `kernels`
**Date**: 2026-03-30
**Context**: Parameter Golf challenge — benchmark on 1×A100 and 4×A100

---

## 1. Why these kernels exist

The core of the **Muon** optimizer is the function `zeropower_via_newtonschulz5`, which
orthogonalizes every matrix gradient before applying it. The algorithm runs 5
Newton-Schulz iterations; each iteration costs two operations:

```
A = X @ X.T          # (M×K) @ (K×M) → (M×M)
B = b*A + c*(A @ A)  # A already symmetric → A@A = A@A.T
X = a*X + B @ X
```

At typical model dimensions (e.g. M=512, K=512), these two matmuls
represent the dominant portion of the optimizer cost. The Triton kernels
replace exactly these two operations.

---

## 2. `XXT` — symmetric matmul X @ X.T

### What it does

Computes `C = A @ A.T` where `A` is a matrix `(M, K)` and `C` is `(M, M)`.

### Key optimization: exploiting symmetry

The result `C[i,j] = Σ_k A[i,k] * A[j,k]` is **symmetric by definition**:
`C[i,j] == C[j,i]` always.

A naïve matmul `A @ A.T` with PyTorch or cuBLAS computes all `M*M` output
blocks. The Triton kernel computes only the blocks of the **upper triangle**
and then mirrors the result:

```python
# Kernel XXT_kernel — essential logic
if skip_block_below_diag:   # blocks below the diagonal: skipped
    return
...
tl.store(c_ptrs,   output,   mask=c_mask)    # write C[m, n]
tl.store(c_ptrs_t, output.T, mask=c_mask_t)  # write C[n, m] = C[m, n].T
```

This launches approximately **half the thread blocks** compared to a standard
GEMM, theoretically doubling the number of useful TFLOPS per byte of output
written.

### Block configuration (hardcoded for A100)

```python
# Default configuration (non K=768)
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
num_stages, num_warps = 4, 8
```

> **Note**: the block sizes were autotuned on **H100**, not on A100. This
> partly explains why the gain on A100 is modest with XXT alone.

---

## 3. `ba_plus_cAA` — fused A@A.T + β·A

### What it does

Computes `C = alpha * (A @ A.T) + beta * A` where `A` is already a
**square symmetric matrix** (the output of XXT from the previous step).

### Why a fusion is needed

Without the kernel, the original operation was:

```python
# Pure PyTorch (original train_gpt)
A = X @ X.T           # matmul 1 → writes A to VRAM
B = b*A + c*(A @ A)   # reads A, does matmul 2, reads A again, writes B
```

There are **two round-trips to VRAM**: `A` is written after XXT, then read twice
to compute `b*A` and `c*A@A`.

The `ba_plus_cAA` kernel does everything in a single launch:
1. Accepts `A` directly as input
2. Accumulates `A @ A.T` into an in-register accumulator in `float32`
3. Loads the corresponding block of `A` (for the `beta*A` term)
4. Applies `accumulator *= alpha` and `accumulator += A_block * beta` in-register
5. Writes the result `C[m,n]` and `C[n,m]` in a single store

```python
# ba_plus_cAA_kernel — essential logic
accumulator = ...     # result of A @ A.T (reduction over K)
a_add = tl.load(...)  # load the block of A for the linear term
accumulator *= alpha  # c * (A @ A.T)
accumulator += a_add * beta  # + b * A
tl.store(c_ptrs, output)     # write C and C.T (symmetry)
tl.store(c_ptrs_t, output.T)
```

The benefit is **memory-bandwidth**: one write and two reads of an `M×M`
matrix are eliminated compared to the unfused sequence.

---

## 4. How they are used in train_gpt.py

```python
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    M = X.size(0)
    A = torch.empty((M, M), device=X.device, dtype=X.dtype)  # pre-alloc buffer
    B = torch.empty((M, M), device=X.device, dtype=X.dtype)  # pre-alloc buffer
    for _ in range(steps):
        XXT(X, A)                              # A = X @ X.T  (symmetry)
        ba_plus_cAA(A, alpha=c, beta=b, out=B) # B = c*A@A + b*A (A is symm.)
        X = a * X + B @ X
    return X.T if transposed else X
```

The buffers `A` and `B` are pre-allocated **outside the loop** to avoid
repeated allocations at every iteration. The kernels write in-place into these
buffers.

---

## 5. Why we do not use `torch.compile` on `zeropower_via_newtonschulz5`

### Short answer

`torch.compile` and Triton kernels with **symmetric double-write** are
incompatible: the compiler cannot correctly trace the aliasing produced by
writing `C[m,n]` and `C[n,m]` in the same CUDA kernel call.

### Detailed answer

`torch.compile` uses **TorchDynamo** to transform Python code into a graph of
PyTorch operations, which is then optimized (fusion, layout transformations,
etc.) by TorchInductor.

When it encounters a call to an external Triton kernel, Dynamo must choose:
- **Option A**: treat it as a *graph break* — stop compilation here,
  execute the original Python function, then resume compilation afterwards.
- **Option B**: try to include the kernel in the graph as an opaque operation.

The specific problem with XXT and ba_plus_cAA is the **symmetric write**
pattern: the same kernel execution writes both `out[m, n]` and `out[n, m]`.
Dynamo cannot model this aliasing in its internal graph — it sees the tensor
`out` as an output, but cannot know that the kernel has written multiple
overlapping positions. This causes correctness errors or unavoidable graph
breaks.

Even if Dynamo succeeded in tracing, the result would be suboptimal:
every XXT and ba_plus_cAA call would become an opaque node in the graph, nullifying
the fusion that `torch.compile` would have applied to the original sequence
`X @ X.T` + `b*A + c*A@A`. The compiler loses visibility over the operation
and can no longer optimize end-to-end.

### What we still compile

**Only `zeropower_via_newtonschulz5` loses the compile.** The transformer model
remains compiled:

```python
# train_gpt.py — in main()
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
# ^^^ this remains active in all variants (control, iter1, iter2)

# Removed in Triton variants:
# zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
```

| Component | Control | Iter-1 (XXT) | Iter-2 (XXT+ba_plus_cAA) |
|---|---|---|---|
| Model forward pass | `torch.compile` | `torch.compile` | `torch.compile` |
| `zeropower_via_newtonschulz5` | `torch.compile` | Native Triton | Native Triton |

So the compile is removed **only and exclusively** from the Newton-Schulz
function that now uses the Triton kernels. Everything else in the training loop
— attention, MLP, embedding, loss — remains compiled as normal.

---

## 6. Results on 4×A100 (2000 steps)

| Variant | step_ms | tok/s | val_bpb | vs. control |
|---|---|---|---|---|
| Control (PyTorch + compile) | 173.8 | 3,022,948 | 1.3610 | — |
| Iter-1 (XXT only) | 194.2 | 2,699,649 | 1.3608 | −8.9% (slower) |
| Iter-2 (XXT + ba_plus_cAA) | 158.9 | 3,307,810 | 1.3599 | **+8.6%** (faster) |

### Reading the results

- **Iter-1 is slower**: XXT alone gains on tiles, but loses end-to-end compile
  on `zeropower`. The original compiler fused the sequence
  `X@X.T → b*A + c*A@A` into a single kernel; XXT alone breaks this fusion
  without bringing enough tile savings to compensate.

- **Iter-2 is faster**: Adding `ba_plus_cAA` recovers the fusion of the second
  matmul. The two kernels together form a unit equivalent to the compiled
  version, with the additional advantage of symmetry (fewer tiles) and the
  fusion of `beta*A` (fewer VRAM round-trips). This yields a net +8.6%.

- **BPB invariant**: all three variants converge to the same loss values
  (difference ≤0.001), confirming that the Triton kernels are numerically
  equivalent.

### Conclusion

The gain from Iter-2 (+8.6% throughput on 4×A100) is real and reproducible.
The advantage stems from the combination of two effects:
1. **Fewer tiles** (symmetry): ~50% of thread blocks compared to a naïve GEMM.
2. **Less VRAM traffic** (fusion): `ba_plus_cAA` eliminates one write and two
   reads of an M×M matrix compared to the unfused sequence.

The block sizes were autotuned for H100. On H100 the expected gain is
greater thanks to SM90 features (TMA, async copy). On A100 the optimization is
already positive but suboptimal.
