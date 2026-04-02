#!/usr/bin/env python3
"""
Generate plots for Round 2 (4×A100) weight-sharing experiment results.

Usage:
    uv run python tools/plot_sharing.py
    uv run python tools/plot_sharing.py --out-dir outputs/plots/sharing_round2

Outputs (saved to --out-dir):
  val_bpb_curves.png   — validation BPB over training steps for all 6 configs
  train_loss_curves.png — training loss over steps for all 6 configs
  bpb_vs_artifact.png  — final int8 BPB vs artifact size scatter/bar
  artifact_savings.png — artifact saving (bytes) vs Δ BPB relative to baseline
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Palette A — project standard
# ---------------------------------------------------------------------------
PALETTE = ["#3D405B", "#E07A5F", "#81B29A", "#F2CC8F", "#F4F1DE", "#9E2A2B"]

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
RUNS_DIR = Path(__file__).parent.parent / "outputs" / "runs"

RUN_CONFIGS = [
    {
        "id": "share4_baseline_20260403_job39005777",
        "label": "Baseline",
        "rv": 0, "ro": 0, "pairs": 0,
    },
    {
        "id": "share4_solo_o_20260403_job39005467",
        "label": "Solo O (ro=256)",
        "rv": 0, "ro": 256, "pairs": 1,
    },
    {
        "id": "share4_solo_v_20260403_job39005468",
        "label": "Solo V (rv=128)",
        "rv": 128, "ro": 0, "pairs": 1,
    },
    {
        "id": "share4_v_o_20260403_job39005469",
        "label": "V+O 128/256",
        "rv": 128, "ro": 256, "pairs": 1,
    },
    {
        "id": "share4_v_o_low_20260403_job39005471",
        "label": "V+O low 64/128",
        "rv": 64, "ro": 128, "pairs": 1,
    },
    {
        "id": "share4_v_multi_20260403_job39005472",
        "label": "V+O multi (4 pairs)",
        "rv": 128, "ro": 256, "pairs": 4,
    },
]

BASELINE_ARTIFACT = 15_768_735


def load_run(run_id: str) -> tuple[list[dict], list[dict], dict | None]:
    """Return (val_events, train_events, final_rt_event)."""
    path = RUNS_DIR / run_id / f"{run_id}.jsonl"
    events = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    val = [e for e in events if e.get("event") == "val_step"]
    train = [e for e in events if e.get("event") == "train_step"]
    final = next((e for e in reversed(events) if e.get("event") == "final_int8_zlib_roundtrip"), None)
    return val, train, final


def get_artifact_bytes(run_id: str) -> int | None:
    for ptz in (RUNS_DIR / run_id).glob("*.int8.ptz"):
        return ptz.stat().st_size
    return None


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
# Plot 1 — Validation BPB curves
# ---------------------------------------------------------------------------
def plot_val_bpb(out_dir: Path) -> None:
    _fig_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cfg, color in zip(RUN_CONFIGS, PALETTE):
        val, _, _ = load_run(cfg["id"])
        steps = [e["step"] for e in val]
        bpb = [e["val_bpb"] for e in val]
        lw = 2.2 if cfg["label"] == "Baseline" else 1.6
        ls = "-" if cfg["label"] == "Baseline" else "-"
        ax.plot(steps, bpb, color=color, lw=lw, ls=ls, label=cfg["label"], marker="o",
                markersize=3.5, markerfacecolor=color, markeredgewidth=0)

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Validation BPB", fontsize=11)
    ax.set_title("Validation BPB — Round 2 (4×A100, ~3 620 steps)", fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    path = out_dir / "val_bpb_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 2 — Training loss curves (sampled every 10 steps for readability)
# ---------------------------------------------------------------------------
def plot_train_loss(out_dir: Path) -> None:
    _fig_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    stride = 10

    for cfg, color in zip(RUN_CONFIGS, PALETTE):
        _, train, _ = load_run(cfg["id"])
        steps = [e["step"] for e in train[::stride]]
        loss = [e["train_loss"] for e in train[::stride]]
        lw = 2.2 if cfg["label"] == "Baseline" else 1.6
        ax.plot(steps, loss, color=color, lw=lw, label=cfg["label"])

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Training loss", fontsize=11)
    ax.set_title("Training loss — Round 2 (4×A100, ~3 620 steps)", fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    path = out_dir / "train_loss_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 3 — Final int8 BPB bar chart
# ---------------------------------------------------------------------------
def plot_bpb_bars(out_dir: Path) -> None:
    _fig_style()
    labels, bpbs, colors = [], [], []

    for cfg, color in zip(RUN_CONFIGS, PALETTE):
        _, _, final = load_run(cfg["id"])
        if final:
            labels.append(cfg["label"])
            bpbs.append(final["val_bpb"])
            colors.append(color)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, bpbs, color=colors, width=0.6, edgecolor="#222222", linewidth=0.6)

    # annotate bars
    for bar, bpb in zip(bars, bpbs):
        ax.text(bar.get_x() + bar.get_width() / 2, bpb + 0.0003,
                f"{bpb:.4f}", ha="center", va="bottom", fontsize=8.5)

    # baseline reference line
    baseline_bpb = bpbs[0]
    ax.axhline(baseline_bpb, color="#9E2A2B", lw=1.2, ls="--", label=f"Baseline {baseline_bpb:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Final int8 BPB", fontsize=11)
    ax.set_title("Final int8 BPB by configuration — Round 2", fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.8)
    y_min = min(bpbs) - 0.002
    y_max = max(bpbs) + 0.004
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    path = out_dir / "final_bpb_bars.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 4 — Artifact savings vs Δ BPB scatter (tradeoff frontier)
# ---------------------------------------------------------------------------
def plot_tradeoff(out_dir: Path) -> None:
    _fig_style()

    _, _, baseline_final = load_run(RUN_CONFIGS[0]["id"])
    baseline_bpb_int8 = baseline_final["val_bpb"]
    baseline_artifact = get_artifact_bytes(RUN_CONFIGS[0]["id"]) or BASELINE_ARTIFACT

    fig, ax = plt.subplots(figsize=(9, 6))

    for cfg, color in zip(RUN_CONFIGS[1:], PALETTE[1:]):  # skip baseline
        _, _, final = load_run(cfg["id"])
        artifact = get_artifact_bytes(cfg["id"])
        if final is None or artifact is None:
            continue
        delta_bpb = final["val_bpb"] - baseline_bpb_int8
        savings_kb = (baseline_artifact - artifact) / 1024
        ax.scatter(savings_kb, delta_bpb, color=color, s=90, zorder=3,
                   edgecolors="#222222", linewidth=0.5)
        ax.annotate(cfg["label"], (savings_kb, delta_bpb),
                    textcoords="offset points", xytext=(7, 3),
                    fontsize=9, color=color)

    # Reference lines
    ax.axhline(0, color="#3D405B", lw=1, ls="--", alpha=0.5, label="Baseline BPB")
    ax.axvline(0, color="#3D405B", lw=1, ls="--", alpha=0.5)

    ax.set_xlabel("Artifact savings (KB vs baseline)", fontsize=11)
    ax.set_ylabel("Δ int8 BPB (vs baseline, lower = better)", fontsize=11)
    ax.set_title("BPB cost vs artifact savings — Round 2 tradeoff frontier", fontsize=12, pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.4f"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f KB"))
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()
    path = out_dir / "tradeoff_frontier.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/plots/sharing_round2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to {out_dir}/")

    plot_val_bpb(out_dir)
    plot_train_loss(out_dir)
    plot_bpb_bars(out_dir)
    plot_tradeoff(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
