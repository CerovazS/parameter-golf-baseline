"""
Benchmark comparison plot for Parameter Golf Triton kernel experiments.
4×A100, 2000 steps — Newton-Schulz Triton kernel variants.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Palette A
# ---------------------------------------------------------------------------
BG      = "#F4F1DE"
RED     = "#E07A5F"
DARK    = "#3D405B"
GREEN   = "#81B29A"
YELLOW  = "#F2CC8F"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
labels = [
    "Control\n(torch.compile)",
    "Iter-1\n(XXT only)*",
    "Iter-2\n(XXT +\nba_plus_cAA)",
    "Iter-3\n(ba_plus_cAA\nonly)",
]

step_ms  = [173.8,  194.2,  158.86, 159.16]
val_bpb  = [1.3610, 1.3608, 1.3599, 1.3629]

colors   = [DARK, RED, GREEN, YELLOW]

# Iter-1 is wall-clock only — flag it
iter1_idx = 1

# % speedup relative to control (negative means slower)
control_ms = step_ms[0]
speedups = [(control_ms - ms) / control_ms * 100 for ms in step_ms]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor(BG)
for ax in (ax_left, ax_right):
    ax.set_facecolor(BG)

x = np.arange(len(labels))
bar_w = 0.55

# ---- Left: step time -------------------------------------------------------
bars_l = []
for i, (xi, ms, col) in enumerate(zip(x, step_ms, colors)):
    hatch = "//" if i == iter1_idx else None
    b = ax_left.bar(xi, ms, width=bar_w, color=col, edgecolor=DARK,
                    linewidth=0.8, hatch=hatch)
    bars_l.append(b)

# Baseline dashed line
ax_left.axhline(control_ms, color=DARK, linestyle="--", linewidth=1.2,
                alpha=0.7, label=f"Control baseline ({control_ms} ms)")

# % speedup annotations
for i, (xi, ms, spd) in enumerate(zip(x, step_ms, speedups)):
    if i == 0:
        ann = "baseline"
        va_offset = ms + 1.5
    else:
        sign = "+" if spd > 0 else ""
        ann = f"{sign}{spd:.1f}%"
        va_offset = ms + 1.5
    color_ann = DARK
    ax_left.text(xi, va_offset, ann, ha="center", va="bottom",
                 fontsize=9, color=color_ann, fontweight="bold")

ax_left.set_xticks(x)
ax_left.set_xticklabels(labels, fontsize=9)
ax_left.set_ylabel("Step time (ms) — GPU compute", fontsize=10, color=DARK)
ax_left.set_title("Step Time (lower = better)", fontsize=11, color=DARK,
                  pad=8)
ax_left.tick_params(colors=DARK)
ax_left.spines[["top", "right"]].set_visible(False)
ax_left.spines[["left", "bottom"]].set_color(DARK)
ax_left.set_ylim(0, max(step_ms) * 1.20)
ax_left.legend(fontsize=8, framealpha=0.5, facecolor=BG, edgecolor=DARK)

# ---- Right: val BPB --------------------------------------------------------
for i, (xi, bpb, col) in enumerate(zip(x, val_bpb, colors)):
    hatch = "//" if i == iter1_idx else None
    ax_right.bar(xi, bpb, width=bar_w, color=col, edgecolor=DARK,
                 linewidth=0.8, hatch=hatch)

# Value annotations above bars
bpb_min = min(val_bpb)
bpb_max = max(val_bpb)
for xi, bpb in zip(x, val_bpb):
    ax_right.text(xi, bpb + 0.0002, f"{bpb:.4f}", ha="center", va="bottom",
                  fontsize=9, color=DARK, fontweight="bold")

ax_right.set_xticks(x)
ax_right.set_xticklabels(labels, fontsize=9)
ax_right.set_ylabel("val BPB @ 2000 steps", fontsize=10, color=DARK)
ax_right.set_title("Validation BPB (lower = better)", fontsize=11, color=DARK,
                   pad=8)
ax_right.tick_params(colors=DARK)
ax_right.spines[["top", "right"]].set_visible(False)
ax_right.spines[["left", "bottom"]].set_color(DARK)
# Tight y-range to show differences clearly
y_pad = (bpb_max - bpb_min) * 5
ax_right.set_ylim(bpb_min - y_pad, bpb_max + y_pad * 3)

# ---- Suptitle & footnote ---------------------------------------------------
fig.suptitle(
    "Triton Newton-Schulz Kernels — 4×A100, 2000 steps",
    fontsize=13, fontweight="bold", color=DARK, y=0.98
)

footnote = (
    "* Iter-1 shows wall-clock step time (meta.json run dir collision). "
    "All others are GPU compute time from meta.json."
)
fig.text(
    0.5, 0.01, footnote,
    ha="center", va="bottom", fontsize=7.5, color=DARK,
    style="italic", wrap=True
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = (
    "/leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf/"
    "reports/kernel_benchmark_4xA100.png"
)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out_path}")
