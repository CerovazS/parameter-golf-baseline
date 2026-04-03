#!/usr/bin/env python3
"""
Generate plots for Round 3 weight-sharing experiments.

Usage:
    uv run python tools/plot_sharing_round3.py
    uv run python tools/plot_sharing_round3.py --out-dir outputs/plots/sharing_round3

Outputs:
  tradeoff_frontier_r3.png  — Δ BPB vs artifact savings scatter, all Round 3 configs
  bpb_bars_r3.png           — Final int8 BPB grouped bar chart by projection family
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Palette A — project standard
# ---------------------------------------------------------------------------
PALETTE = {
    "baseline":  "#3D405B",
    "round2_ref": "#9E2A2B",
    "Q":         "#81B29A",
    "O":         "#E07A5F",
    "MLP":       "#F2CC8F",
    "combo":     "#F4F1DE",
}
COLOR_MAP = {
    "Q":    "#81B29A",
    "O":    "#E07A5F",
    "MLP":  "#F2CC8F",
    "O+Q":  "#3D405B",
    "combo":"#9E2A2B",
}

BASELINE_BPB    = 1.2643
BASELINE_ART    = 15_768_735
CEILING         = 16_000_000
HEADROOM        = CEILING - BASELINE_ART  # 231 265

# ---------------------------------------------------------------------------
# Data — hardcoded from Round 3 final metrics
# ---------------------------------------------------------------------------
ROUND2_REF = [
    {"label": "Solo O ro=256 p=1\n(Round 2)", "family": "O",
     "bpb": 1.2668, "artifact": 15_634_110, "ref": True},
    {"label": "V+O low 64/128 p=1\n(Round 2)", "family": "O",
     "bpb": 1.2684, "artifact": 15_517_908, "ref": True},
]

ROUND3 = [
    # --- Q ---
    {"label": "Q r=256 p=1",   "family": "Q",   "bpb": 1.2640, "artifact": 15_651_216},
    {"label": "Q r=256 p=2",   "family": "Q",   "bpb": 1.2668, "artifact": 15_405_243},
    # --- O ---
    {"label": "O r=256 p=2",   "family": "O",   "bpb": 1.2676, "artifact": 15_388_314},
    {"label": "O r=256 p=3",   "family": "O",   "bpb": 1.2713, "artifact": 15_154_897},
    {"label": "O r=256 p=4",   "family": "O",   "bpb": 1.2709, "artifact": 14_923_209},
    # --- MLP ---
    {"label": "MLP_FC r=256 p=1",   "family": "MLP", "bpb": 1.2715, "artifact": 15_399_079},
    {"label": "MLP_PROJ r=256 p=1", "family": "MLP", "bpb": 1.2711, "artifact": 15_269_849},
    {"label": "MLP_both r=256 p=1", "family": "MLP", "bpb": 1.2793, "artifact": 14_924_042},
    {"label": "MLP_FC r=256 p=2",   "family": "MLP", "bpb": 1.2762, "artifact": 14_901_022},
    # --- Combinations ---
    {"label": "O+Q r=256 p=1", "family": "O+Q", "bpb": 1.2666, "artifact": 15_520_697},
    {"label": "O+Q r=256 p=2", "family": "O+Q", "bpb": 1.2702, "artifact": 15_038_260},
    {"label": "O+MLP_FC r=256 p=1", "family": "combo", "bpb": 1.2727, "artifact": 15_273_737},
    # full failed — excluded from frontier (1163 steps only)
]

BAR_GROUPS = [
    ("Q", [d for d in ROUND3 if d["family"] == "Q"]),
    ("O", [d for d in ROUND3 if d["family"] == "O"]),
    ("MLP", [d for d in ROUND3 if d["family"] == "MLP"]),
    ("O+Q / combo", [d for d in ROUND3 if d["family"] in ("O+Q", "combo")]),
]


def _fig_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Plot 1 — Tradeoff frontier
# ---------------------------------------------------------------------------
def plot_tradeoff(out_dir: Path) -> None:
    _fig_style()
    fig, ax = plt.subplots(figsize=(11, 7))

    # Round 2 reference points (muted, dashed outlines)
    for cfg in ROUND2_REF:
        dx = (BASELINE_ART - cfg["artifact"]) / 1024
        dy = cfg["bpb"] - BASELINE_BPB
        ax.scatter(dx, dy, color="#9E2A2B", s=80, marker="D",
                   edgecolors="#222222", linewidth=0.6, alpha=0.55, zorder=3)
        ax.annotate(cfg["label"], (dx, dy),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=7.5, color="#9E2A2B", alpha=0.75)

    # Round 3 points
    family_color = {
        "Q": COLOR_MAP["Q"],
        "O": COLOR_MAP["O"],
        "MLP": COLOR_MAP["MLP"],
        "O+Q": COLOR_MAP["O+Q"],
        "combo": COLOR_MAP["combo"],
    }
    for cfg in ROUND3:
        dx = (BASELINE_ART - cfg["artifact"]) / 1024
        dy = cfg["bpb"] - BASELINE_BPB
        c = family_color[cfg["family"]]
        ax.scatter(dx, dy, color=c, s=95, edgecolors="#222222",
                   linewidth=0.6, zorder=4)
        ax.annotate(cfg["label"], (dx, dy),
                    textcoords="offset points", xytext=(7, 3),
                    fontsize=8, color=c)

    # Baseline marker
    ax.scatter(0, 0, color=PALETTE["baseline"], s=110, marker="*",
               edgecolors="#222222", linewidth=0.6, zorder=5,
               label=f"Baseline  BPB={BASELINE_BPB:.4f}")

    # Reference lines
    ax.axhline(0, color=PALETTE["baseline"], lw=1, ls="--", alpha=0.45)
    ax.axvline(0, color=PALETTE["baseline"], lw=1, ls="--", alpha=0.45)
    ax.axvline(HEADROOM / 1024, color="#E07A5F", lw=1.2, ls=":",
               alpha=0.7, label=f"Baseline headroom ({HEADROOM/1024:.0f} KB)")

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=family_color[f], label=f)
               for f in ("Q", "O", "MLP", "O+Q", "combo")]
    ref_patch = mpatches.Patch(color="#9E2A2B", alpha=0.55, label="Round 2 ref")
    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches + [ref_patch],
              fontsize=8.5, framealpha=0.85, loc="upper left")

    ax.set_xlabel("Artifact savings vs baseline (KB)", fontsize=11)
    ax.set_ylabel("Δ int8 BPB vs baseline", fontsize=11)
    ax.set_title(
        "Round 3 — BPB/artifact tradeoff frontier\n"
        "Weight-sharing sweep: Q, O multi-pair, MLP, combinations",
        fontsize=12, pad=10,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.4f"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f KB"))
    fig.tight_layout()
    path = out_dir / "tradeoff_frontier_r3.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 2 — BPB bar chart grouped by family
# ---------------------------------------------------------------------------
def plot_bpb_bars(out_dir: Path) -> None:
    _fig_style()

    all_labels, all_bpbs, all_colors = [], [], []
    sep_positions = []
    pos = 0
    group_centers = []
    group_names = []

    for g_name, items in BAR_GROUPS:
        group_start = pos
        for cfg in items:
            all_labels.append(cfg["label"].replace(" r=256", "").replace("\n", " "))
            all_bpbs.append(cfg["bpb"])
            all_colors.append(family_color_for(cfg["family"]))
            pos += 1
        sep_positions.append(pos + 0.3)
        group_centers.append((group_start + pos - 1) / 2)
        group_names.append(g_name)
        pos += 1  # gap between groups

    x = np.arange(len(all_labels))
    x_mapped = []
    idx = 0
    for g_name, items in BAR_GROUPS:
        for _ in items:
            x_mapped.append(idx)
            idx += 1
        idx += 1  # gap

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(x_mapped, all_bpbs, color=all_colors, width=0.7,
                  edgecolor="#333333", linewidth=0.5)

    for bar, bpb in zip(bars, all_bpbs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bpb + 0.0003, f"{bpb:.4f}",
                ha="center", va="bottom", fontsize=7)

    ax.axhline(BASELINE_BPB, color="#3D405B", lw=1.4, ls="--",
               label=f"Baseline {BASELINE_BPB:.4f}")

    # Group separators and labels
    for sep in sep_positions[:-1]:
        ax.axvline(sep, color="#aaaaaa", lw=0.8, ls="-", alpha=0.5)
    for center, name in zip(group_centers, group_names):
        ax.text(center, BASELINE_BPB - 0.0015, name,
                ha="center", va="top", fontsize=9,
                color="#333333", fontweight="bold")

    ax.set_xticks(x_mapped)
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Final int8 BPB", fontsize=11)
    ax.set_title("Round 3 — Final int8 BPB by configuration", fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.85)
    y_min = min(all_bpbs) - 0.003
    y_max = max(all_bpbs) + 0.008
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    path = out_dir / "bpb_bars_r3.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def family_color_for(family: str) -> str:
    return COLOR_MAP.get(family, "#cccccc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/plots/sharing_round3")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to {out_dir}/")

    plot_tradeoff(out_dir)
    plot_bpb_bars(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
