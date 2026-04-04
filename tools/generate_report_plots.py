#!/usr/bin/env python3
"""Generate publication-quality plots for Spectron low-rank architecture sweep experiments.

Reads meta.json from all run directories under outputs/runs/ and produces six
analysis plots saved to outputs/report/plots/.

Palette B: #335C67, #FFF3B0, #E09F3E, #9E2A2B, #540B0E
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Palette B
# ---------------------------------------------------------------------------
P = {
    "teal": "#335C67",
    "cream": "#FFF3B0",
    "amber": "#E09F3E",
    "crimson": "#9E2A2B",
    "dark": "#540B0E",
}
PALETTE_ORDERED = [P["teal"], P["amber"], P["crimson"], P["dark"], P["cream"]]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT / "outputs" / "runs"
OUT_DIR = PROJECT / "outputs" / "report" / "plots"

BASELINE_BPB = 1.2612  # dense baseline val BPB (from meta.json)
BUDGET_BYTES = 16_000_000  # 16 MB artifact limit


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class Run:
    run_id: str
    category: str  # parent dir name e.g. "02_lr_sweep"
    status: str
    hypothesis: str = ""
    train_script: str = ""

    # results
    final_val_bpb: Optional[float] = None
    final_int8_bpb: Optional[float] = None
    artifact_bytes: Optional[int] = None
    total_steps: Optional[int] = None
    mean_step_ms: Optional[float] = None
    mean_tok_s: Optional[float] = None

    # hyperparameters
    model_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    mlp_mult: Optional[int] = None
    matrix_lr: Optional[float] = None
    rank_ratio: Optional[float] = None
    head_dim: Optional[int] = None

    @property
    def artifact_mb(self) -> float:
        return (self.artifact_bytes or 0) / 1e6

    @property
    def label_short(self) -> str:
        """Short config label like D512 L9 or D768 L8 nkv2."""
        parts = [f"D{self.model_dim}", f"L{self.num_layers}"]
        if self.mlp_mult and self.mlp_mult != 2:
            parts.append(f"mlp{self.mlp_mult}")
        return " ".join(parts)

    @property
    def label_dx_ly(self) -> str:
        return f"D{self.model_dim}xL{self.num_layers}"

    @property
    def is_baseline(self) -> bool:
        return self.train_script == "train_gpt.py"

    @property
    def gqa_ratio_str(self) -> str:
        if self.num_heads and self.num_kv_heads:
            r = self.num_heads // self.num_kv_heads
            if self.num_kv_heads == self.num_heads:
                return f"{r}:1 MHA"
            return f"{r}:1"
        return "?"


def _parse_rank_ratio(hypothesis: str, run_id: str) -> Optional[float]:
    """Extract rank_ratio from hypothesis string or run_id naming convention."""
    m = re.search(r"rank_ratio=(\d+\.?\d*)", hypothesis)
    if m:
        return float(m.group(1))
    m = re.search(r"_r(\d{3})_", run_id)
    if m:
        return int(m.group(1)) / 100.0
    return None


def load_runs() -> list[Run]:
    """Walk outputs/runs/<category>/<run_id>/meta.json and return parsed Run objects."""
    runs: list[Run] = []
    for cat_dir in sorted(RUNS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for run_dir in sorted(cat_dir.iterdir()):
            meta_path = run_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                d = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            r = d.get("results") or {}
            h = d.get("hyperparameters") or {}
            hypothesis = d.get("hypothesis", "")
            run_id = d.get("run_id", run_dir.name)

            run = Run(
                run_id=run_id,
                category=cat_dir.name,
                status=d.get("status", "unknown"),
                hypothesis=hypothesis,
                train_script=d.get("train_script", ""),
                final_val_bpb=r.get("final_val_bpb"),
                final_int8_bpb=r.get("final_int8_bpb"),
                artifact_bytes=r.get("artifact_bytes"),
                total_steps=r.get("total_steps"),
                mean_step_ms=r.get("mean_step_ms"),
                mean_tok_s=r.get("mean_tok_s"),
                model_dim=h.get("MODEL_DIM"),
                num_layers=h.get("NUM_LAYERS"),
                num_heads=h.get("NUM_HEADS"),
                num_kv_heads=h.get("NUM_KV_HEADS"),
                mlp_mult=h.get("MLP_MULT"),
                matrix_lr=h.get("MATRIX_LR"),
                rank_ratio=_parse_rank_ratio(hypothesis, run_id),
                head_dim=h.get("HEAD_DIM"),
            )
            runs.append(run)
    return runs


def filter_valid(runs: list[Run]) -> list[Run]:
    """Keep only completed runs with BPB <= 3.0 (exclude failed and diverged)."""
    out = []
    for r in runs:
        if r.status != "completed":
            continue
        if r.final_val_bpb is None or r.final_val_bpb > 3.0:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Category coloring
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    "01_baseline": P["dark"],
    "02_lr_sweep": P["teal"],
    "03_depth_scaling": P["amber"],
    "04_width_scaling": P["crimson"],
    "05_gqa_sweep": P["teal"],
    "06_mlp_width": P["amber"],
    "07_head_dim": P["crimson"],
    "08_per_layer": P["dark"],
}

CATEGORY_LABELS = {
    "01_baseline": "Dense Baseline",
    "02_lr_sweep": "LR Sweep",
    "03_depth_scaling": "Depth Scaling",
    "04_width_scaling": "Width Scaling",
    "05_gqa_sweep": "GQA Sweep",
    "06_mlp_width": "MLP Width",
    "07_head_dim": "Head Dim",
    "08_per_layer": "Per-Layer",
}


def cat_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "#888888")


def cat_label(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat)


# ---------------------------------------------------------------------------
# Common figure setup
# ---------------------------------------------------------------------------
def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.alpha": 0.4,
    })


def save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Plot 1: BPB Ranking (horizontal bar chart)
# ---------------------------------------------------------------------------
def plot_01_bpb_ranking(runs: list[Run]):
    runs_sorted = sorted(runs, key=lambda r: r.final_val_bpb, reverse=True)

    fig, ax1 = plt.subplots(figsize=(11, max(6, len(runs_sorted) * 0.32)))

    y_pos = np.arange(len(runs_sorted))
    bpb_vals = [r.final_val_bpb for r in runs_sorted]
    colors = [cat_color(r.category) for r in runs_sorted]
    labels = []
    for r in runs_sorted:
        lbl = r.label_short
        if r.is_baseline:
            lbl += " (dense)"
        elif r.rank_ratio is not None:
            lbl += f" rr={r.rank_ratio:.2f}"
        if r.matrix_lr:
            lbl += f" lr={r.matrix_lr}"
        labels.append(lbl)

    bars = ax1.barh(y_pos, bpb_vals, color=colors, edgecolor="white", linewidth=0.5,
                    height=0.7, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Validation BPB (lower is better)")
    ax1.set_title("Spectron Architecture Sweep: BPB Ranking")
    ax1.invert_yaxis()

    # Baseline reference line
    ax1.axvline(BASELINE_BPB, color=P["dark"], linestyle="--", linewidth=1.5,
                label=f"Dense baseline ({BASELINE_BPB:.4f})", zorder=5)

    # Artifact size annotations on right
    for i, r in enumerate(runs_sorted):
        mb = r.artifact_mb
        marker = " *" if mb > BUDGET_BYTES / 1e6 else ""
        ax1.annotate(f"{mb:.1f} MB{marker}",
                     xy=(r.final_val_bpb, i),
                     xytext=(4, 0), textcoords="offset points",
                     fontsize=7, va="center", color="#555555")

    # Budget annotation
    ax1.annotate(f"16 MB budget limit", xy=(0.98, 0.02), xycoords="axes fraction",
                 fontsize=8, ha="right", va="bottom", color=P["crimson"],
                 fontstyle="italic")
    ax1.annotate("* = over budget", xy=(0.98, 0.06), xycoords="axes fraction",
                 fontsize=7, ha="right", va="bottom", color="#888888",
                 fontstyle="italic")

    # BPB range
    min_bpb = min(bpb_vals)
    max_bpb = max(bpb_vals)
    ax1.set_xlim(min_bpb - 0.01, max_bpb + 0.035)

    # Legend for categories
    from matplotlib.patches import Patch
    seen = {}
    for r in runs_sorted:
        if r.category not in seen:
            seen[r.category] = cat_color(r.category)
    legend_elements = [Patch(facecolor=c, label=cat_label(k)) for k, c in seen.items()]
    legend_elements.append(plt.Line2D([0], [0], color=P["dark"], linestyle="--",
                                       linewidth=1.5, label=f"Dense baseline"))
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8,
               framealpha=0.9)

    fig.tight_layout()
    save(fig, "01_bpb_ranking.png")


# ---------------------------------------------------------------------------
# Plot 2: Width vs Depth scatter (artifact MB vs BPB)
# ---------------------------------------------------------------------------
def plot_02_width_vs_depth(runs: list[Run]):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by model_dim
    dims = sorted(set(r.model_dim for r in runs if r.model_dim))
    dim_colors = {}
    color_cycle = [P["teal"], P["amber"], P["crimson"], P["dark"], P["cream"]]
    for i, d in enumerate(dims):
        dim_colors[d] = color_cycle[i % len(color_cycle)]

    for r in runs:
        mb = r.artifact_mb
        bpb = r.final_val_bpb
        c = dim_colors.get(r.model_dim, "#888888")
        marker = "D" if r.is_baseline else "o"
        size = 80 if r.is_baseline else 50
        ax.scatter(mb, bpb, c=c, s=size, marker=marker, edgecolors="white",
                   linewidths=0.5, zorder=3, alpha=0.85)
        ax.annotate(r.label_dx_ly, (mb, bpb), fontsize=6.5,
                    textcoords="offset points", xytext=(5, 3), color="#444444")

    # Baseline horizontal line
    ax.axhline(BASELINE_BPB, color=P["dark"], linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Dense baseline BPB ({BASELINE_BPB:.4f})")

    # Budget vertical line
    ax.axvline(BUDGET_BYTES / 1e6, color=P["crimson"], linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"16 MB budget")

    # Shade over-budget region
    xlim = ax.get_xlim()
    ax.axvspan(BUDGET_BYTES / 1e6, xlim[1] + 5, alpha=0.06, color=P["crimson"],
               zorder=0)

    ax.set_xlabel("Compressed Artifact Size (MB)")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title("Width vs Depth: Artifact Size vs Quality (Pareto View)")

    # Legend for dims
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=dim_colors[d],
                       markersize=8, label=f"D={d}") for d in dims]
    handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor=P["dark"],
                          markersize=8, label="Dense baseline"))
    handles.append(Line2D([0], [0], color=P["dark"], linestyle="--", label="Baseline BPB"))
    handles.append(Line2D([0], [0], color=P["crimson"], linestyle="--", label="16 MB budget"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save(fig, "02_width_vs_depth.png")


# ---------------------------------------------------------------------------
# Plot 3: Depth Scaling for D=512 at lr=0.08
# ---------------------------------------------------------------------------
def plot_03_depth_scaling_d512(runs: list[Run]):
    # Filter: D=512, lr=0.08 (close to), and from categories likely to have depth variation
    subset = [r for r in runs
              if r.model_dim == 512
              and r.matrix_lr is not None
              and abs(r.matrix_lr - 0.08) < 0.005]

    if not subset:
        print("  WARN: No D=512 lr=0.08 runs found for plot 03")
        return

    # Group by num_layers, pick best BPB per layer count
    by_layers: dict[int, list[Run]] = {}
    for r in subset:
        by_layers.setdefault(r.num_layers, []).append(r)

    layers_sorted = sorted(by_layers.keys())
    best_bpb = []
    best_steps = []
    all_bpb = []  # for scatter showing all points
    all_layers = []
    all_tok_s = []

    for nl in layers_sorted:
        group = by_layers[nl]
        best = min(group, key=lambda r: r.final_val_bpb)
        best_bpb.append(best.final_val_bpb)
        best_steps.append(best.total_steps or 0)
        for r in group:
            all_layers.append(nl)
            all_bpb.append(r.final_val_bpb)
            all_tok_s.append(r.mean_tok_s or 0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Line for best BPB
    ax1.plot(layers_sorted, best_bpb, "o-", color=P["teal"], linewidth=2.2,
             markersize=8, label="Best BPB", zorder=4)

    # Scatter all points (fainter)
    ax1.scatter(all_layers, all_bpb, color=P["teal"], s=25, alpha=0.35, zorder=3,
                label="All runs")

    # Baseline line
    ax1.axhline(BASELINE_BPB, color=P["dark"], linestyle="--", linewidth=1.2,
                alpha=0.7, label=f"Dense baseline ({BASELINE_BPB:.4f})")

    # Tokens/sec on secondary axis (from best runs)
    best_tok_s = []
    for nl in layers_sorted:
        best = min(by_layers[nl], key=lambda r: r.final_val_bpb)
        best_tok_s.append((best.mean_tok_s or 0) / 1e6)  # Mtok/s

    ax2.bar(layers_sorted, best_tok_s, alpha=0.25, color=P["amber"], width=0.6,
            label="Throughput (best run)", zorder=1)
    ax2.set_ylabel("Throughput (M tok/s)", color=P["amber"])
    ax2.tick_params(axis="y", labelcolor=P["amber"])

    ax1.set_xlabel("Number of Layers")
    ax1.set_ylabel("Validation BPB (lower is better)")
    ax1.set_title("Depth Scaling: D=512, lr=0.08 (Depth-Throughput Tradeoff)")
    ax1.set_xticks(layers_sorted)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8,
               framealpha=0.9)

    fig.tight_layout()
    save(fig, "03_depth_scaling_d512.png")


# ---------------------------------------------------------------------------
# Plot 4: GQA Sweep for D=768 L=8
# ---------------------------------------------------------------------------
def plot_04_gqa_sweep(runs: list[Run]):
    # Filter: D=768, L=8, from GQA sweep or width scaling with matching config
    subset = [r for r in runs
              if r.model_dim == 768
              and r.num_layers == 8
              and r.num_heads == 12]

    if not subset:
        print("  WARN: No D=768 L=8 runs found for plot 04")
        return

    # Group by nkv, pick best per group
    by_nkv: dict[int, list[Run]] = {}
    for r in subset:
        by_nkv.setdefault(r.num_kv_heads, []).append(r)

    nkv_sorted = sorted(by_nkv.keys())
    best_runs = [min(by_nkv[nkv], key=lambda r: r.final_val_bpb) for nkv in nkv_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(nkv_sorted))
    bpb_vals = [r.final_val_bpb for r in best_runs]
    colors_bar = [PALETTE_ORDERED[i % len(PALETTE_ORDERED)] for i in range(len(nkv_sorted))]

    bars = ax.bar(x_pos, bpb_vals, color=colors_bar, edgecolor="white", linewidth=0.8,
                  width=0.6, alpha=0.9, zorder=3)

    # Labels
    gqa_labels = []
    for nkv in nkv_sorted:
        nh = 12  # all have nh=12
        ratio = nh // nkv
        if nkv == nh:
            gqa_labels.append(f"nkv={nkv}\n{ratio}:1 MHA")
        else:
            gqa_labels.append(f"nkv={nkv}\n{ratio}:1 GQA")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(gqa_labels, fontsize=9)

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, bpb_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#333333")

    # Artifact size inside bars
    for i, (bar, r) in enumerate(zip(bars, best_runs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.005,
                f"{r.artifact_mb:.1f} MB", ha="center", va="top", fontsize=7,
                color="white", fontweight="bold")

    # Baseline line
    ax.axhline(BASELINE_BPB, color=P["dark"], linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Dense baseline ({BASELINE_BPB:.4f})")

    # Adjust y-axis range to zoom in on differences
    min_bpb = min(bpb_vals)
    max_bpb = max(bpb_vals)
    margin = (max_bpb - min_bpb) * 0.4
    y_low = min(min_bpb - margin, BASELINE_BPB - 0.01)
    y_high = max_bpb + margin
    ax.set_ylim(y_low, y_high)

    ax.set_xlabel("KV Head Configuration (num_heads=12)")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title("GQA Sweep: D=768, L=8, nh=12 (Effect of KV Head Count)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    save(fig, "04_gqa_sweep.png")


# ---------------------------------------------------------------------------
# Plot 5: Width Scaling for L=9 at lr=0.08
# ---------------------------------------------------------------------------
def plot_05_width_scaling(runs: list[Run]):
    subset = [r for r in runs
              if r.num_layers == 9
              and r.matrix_lr is not None
              and abs(r.matrix_lr - 0.08) < 0.005]

    if not subset:
        print("  WARN: No L=9 lr=0.08 runs found for plot 05")
        return

    # Group by rank_ratio
    by_rr: dict[Optional[float], list[Run]] = {}
    for r in subset:
        by_rr.setdefault(r.rank_ratio, []).append(r)

    fig, ax = plt.subplots(figsize=(10, 6))

    rr_colors = {
        None: P["dark"],
        0.25: P["teal"],
        0.30: P["amber"],
        0.40: P["crimson"],
    }

    for rr, group in sorted(by_rr.items(), key=lambda x: (x[0] is None, x[0] or 0)):
        # Group by model_dim, pick best per dim
        by_dim: dict[int, list[Run]] = {}
        for r in group:
            by_dim.setdefault(r.model_dim, []).append(r)

        dims_sorted = sorted(by_dim.keys())
        best_bpb = [min(by_dim[d], key=lambda r: r.final_val_bpb).final_val_bpb
                     for d in dims_sorted]

        label = "Dense (no rank)" if rr is None else f"rr={rr:.2f}"
        color = rr_colors.get(rr, "#888888")
        marker = "D" if rr is None else "o"

        ax.plot(dims_sorted, best_bpb, f"-", color=color, marker=marker,
                linewidth=2, markersize=8, label=label, zorder=3)

        # Annotate points
        for d, b in zip(dims_sorted, best_bpb):
            ax.annotate(f"D{d}", (d, b), textcoords="offset points",
                       xytext=(5, 5), fontsize=7, color="#555555")

    # Baseline line
    ax.axhline(BASELINE_BPB, color=P["dark"], linestyle="--", linewidth=1.0,
               alpha=0.5, label=f"Dense baseline ({BASELINE_BPB:.4f})")

    ax.set_xlabel("Model Dimension (D)")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title("Width Scaling: L=9, lr=0.08 (Effect of Model Width)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    save(fig, "05_width_scaling.png")


# ---------------------------------------------------------------------------
# Plot 6: Throughput vs Quality scatter
# ---------------------------------------------------------------------------
def plot_06_throughput_vs_quality(runs: list[Run]):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Point size proportional to artifact MB
    max_mb = max(r.artifact_mb for r in runs)

    for r in runs:
        tok_s = (r.mean_tok_s or 0) / 1e6  # M tok/s
        bpb = r.final_val_bpb
        mb = r.artifact_mb
        size = 30 + 150 * (mb / max_mb)
        c = cat_color(r.category)
        marker = "D" if r.is_baseline else "o"

        ax.scatter(tok_s, bpb, c=c, s=size, marker=marker, edgecolors="white",
                   linewidths=0.4, alpha=0.8, zorder=3)

    # Annotate key configs: best BPB, fastest, baseline, and best under budget
    under_budget = [r for r in runs if r.artifact_bytes and r.artifact_bytes <= BUDGET_BYTES]
    key_runs = set()

    # Best overall BPB
    best_bpb = min(runs, key=lambda r: r.final_val_bpb)
    key_runs.add(best_bpb.run_id)

    # Fastest throughput
    fastest = max(runs, key=lambda r: r.mean_tok_s or 0)
    key_runs.add(fastest.run_id)

    # Baseline
    baselines = [r for r in runs if r.is_baseline]
    if baselines:
        key_runs.add(baselines[0].run_id)

    # Best under budget
    if under_budget:
        best_ub = min(under_budget, key=lambda r: r.final_val_bpb)
        key_runs.add(best_ub.run_id)

    for r in runs:
        if r.run_id in key_runs:
            tok_s = (r.mean_tok_s or 0) / 1e6
            bpb = r.final_val_bpb
            label = r.label_short
            if r.is_baseline:
                label += " (dense)"
            ax.annotate(label, (tok_s, bpb), fontsize=7.5,
                       textcoords="offset points", xytext=(8, -5),
                       fontweight="bold", color="#333333",
                       arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                       lw=0.5))

    # Baseline horizontal
    ax.axhline(BASELINE_BPB, color=P["dark"], linestyle="--", linewidth=1.0,
               alpha=0.5, label=f"Dense baseline BPB ({BASELINE_BPB:.4f})")

    ax.set_xlabel("Throughput (M tok/s)")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title("Throughput vs Quality: All Successful Runs")

    # Legend for categories + size reference
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    seen = {}
    for r in runs:
        if r.category not in seen:
            seen[r.category] = cat_color(r.category)
    handles = [Patch(facecolor=c, label=cat_label(k)) for k, c in seen.items()]
    handles.append(Line2D([0], [0], color=P["dark"], linestyle="--",
                          label="Dense baseline"))

    # Size legend
    for ref_mb in [8, 12, 16]:
        s = 30 + 150 * (ref_mb / max_mb)
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#cccccc", markersize=np.sqrt(s),
                              label=f"{ref_mb} MB"))

    ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.9,
              ncol=2)

    fig.tight_layout()
    save(fig, "06_throughput_vs_quality.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading runs...")
    all_runs = load_runs()
    print(f"  Total runs found: {len(all_runs)}")

    runs = filter_valid(all_runs)
    print(f"  Valid runs (completed, BPB <= 3.0): {len(runs)}")

    failed = [r for r in all_runs if r.status == "failed"]
    diverged = [r for r in all_runs if r.status == "completed"
                and r.final_val_bpb is not None and r.final_val_bpb > 3.0]
    print(f"  Skipped: {len(failed)} failed, {len(diverged)} diverged")
    print()

    print("Generating plots...")
    plot_01_bpb_ranking(runs)
    plot_02_width_vs_depth(runs)
    plot_03_depth_scaling_d512(runs)
    plot_04_gqa_sweep(runs)
    plot_05_width_scaling(runs)
    plot_06_throughput_vs_quality(runs)

    print()
    print(f"All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
