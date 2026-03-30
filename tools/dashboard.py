#!/usr/bin/env python3
"""
Parameter Golf Dashboard — self-contained interactive HTML report.

Scans ``outputs/`` for experiment runs (directories with ``.jsonl`` files),
reads metrics and optional ``meta.json``, and produces a single interactive
HTML dashboard with Plotly charts, run selection, and sortable tables.

Usage::

    python tools/dashboard.py                          # all discovered runs
    python tools/dashboard.py --runs run_a run_b       # specific runs only
    python tools/dashboard.py --baseline <run_id>      # explicit baseline
    python tools/dashboard.py -o report.html           # custom output path
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]
except ImportError:
    print("plotly is required:  uv add plotly  or  pip install plotly", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Palette — expanded for visual distinction across many runs
# ---------------------------------------------------------------------------
RUN_PALETTE = [
    "#E09F3E",  # amber
    "#9E2A2B",  # crimson
    "#81B29A",  # sage
    "#E07A5F",  # terra cotta
    "#6D597A",  # mauve
    "#B56576",  # rose
    "#355070",  # steel blue
    "#EAAC8B",  # peach
    "#56CFE1",  # cyan
    "#A7C957",  # lime
    "#F4845F",  # coral
    "#7209B7",  # violet
    "#4CC9F0",  # sky
    "#F72585",  # magenta
    "#4895EF",  # blue
]
BASELINE_COLOR = "#335C67"

# UI theme
BG_PAGE = "#0f0f17"
BG_CARD = "#16182a"
BG_CARD_HOVER = "#1c1f35"
BORDER = "#252840"
TEXT_PRIMARY = "#e8e8ed"
TEXT_SECONDARY = "#8b8da0"
TEXT_MUTED = "#555770"
ACCENT = "#E09F3E"
GREEN = "#4ade80"
RED = "#f87171"

# ---------------------------------------------------------------------------
# Data loading (unchanged logic)
# ---------------------------------------------------------------------------

def discover_runs(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    runs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if list(root.glob("*.jsonl")):
            real = root.resolve()
            if real not in seen:
                seen.add(real)
                runs.append(root)
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and list(d.glob("*.jsonl")):
                real = d.resolve()
                if real not in seen:
                    seen.add(real)
                    runs.append(d)
    return runs


def load_run(run_dir: Path) -> dict | None:
    jsonl_files = sorted(run_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    events: list[dict] = []
    for jf in jsonl_files:
        for line in jf.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"run_id": run_dir.name, "hypothesis": "", "status": "unknown"}

    train_steps = [e for e in events if e.get("event") == "train_step"]
    val_steps = [e for e in events if e.get("event") == "val_step"]
    final_rt = [e for e in events if e.get("event") == "final_int8_zlib_roundtrip"]

    if not meta.get("results"):
        results: dict[str, object] = {}
        if val_steps:
            results["final_val_bpb"] = val_steps[-1].get("val_bpb")
            results["final_val_loss"] = val_steps[-1].get("val_loss")
        if final_rt:
            results["final_int8_bpb"] = final_rt[-1].get("val_bpb")
            results["final_int8_loss"] = final_rt[-1].get("val_loss")
        if train_steps:
            results["total_steps"] = train_steps[-1].get("step", 0)
            results["total_train_time_ms"] = train_steps[-1].get("train_time_ms", 0)
            tok_vals = [e["tok_s"] for e in train_steps if e.get("tok_s")]
            if tok_vals:
                results["mean_tok_s"] = round(sum(tok_vals) / len(tok_vals), 1)
                results["peak_tok_s"] = round(max(tok_vals), 1)
            step_ms = [e["step_time_ms"] for e in train_steps if e.get("step_time_ms")]
            if step_ms:
                results["mean_step_ms"] = round(sum(step_ms) / len(step_ms), 2)
        for ptz in run_dir.glob("*.int8.ptz"):
            results["artifact_bytes"] = ptz.stat().st_size
        for pt in run_dir.glob("final_model.pt"):
            results["model_bytes"] = pt.stat().st_size
        meta["results"] = results

    return {
        "meta": meta,
        "train_steps": train_steps,
        "val_steps": val_steps,
        "final_rt": final_rt,
        "dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Color assignment
# ---------------------------------------------------------------------------

def assign_colors(runs: list[dict], baseline_id: str | None) -> dict[str, str]:
    """Assign a distinct color to each run. Baseline gets its own color."""
    mapping: dict[str, str] = {}
    palette_idx = 0
    for run in runs:
        rid = run["meta"]["run_id"]
        if rid == baseline_id:
            mapping[rid] = BASELINE_COLOR
        else:
            mapping[rid] = RUN_PALETTE[palette_idx % len(RUN_PALETTE)]
            palette_idx += 1
    return mapping


# ---------------------------------------------------------------------------
# Chart builders — return (fig_json, trace_run_map)
# Each fig_json is the Plotly figure serialized as a dict.
# trace_run_map = {run_id: [trace_indices]} for visibility toggling.
# ---------------------------------------------------------------------------

def _chart_layout(height: int = 520, **kw) -> dict:
    layout = dict(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,23,0.6)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT_PRIMARY),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=56, r=24, t=36, b=64),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        hoverlabel=dict(
            bgcolor=BG_CARD, bordercolor=BORDER,
            font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT_PRIMARY),
        ),
    )
    layout.update(kw)
    return layout


def build_train_loss_fig(runs: list[dict], colors: dict[str, str]) -> tuple[str, dict]:
    fig = go.Figure()
    trace_map: dict[str, list[int]] = {}
    idx = 0
    for run in runs:
        rid = run["meta"]["run_id"]
        color = colors[rid]
        steps = [e["step"] for e in run["train_steps"]]
        losses = [e["train_loss"] for e in run["train_steps"]]
        if not steps:
            continue
        fig.add_trace(go.Scattergl(
            x=steps, y=losses, name=rid,
            line=dict(color=color, width=2),
            hovertemplate="<b>%{fullData.name}</b><br>step %{x:,}<br>loss %{y:.4f}<extra></extra>",
        ))
        trace_map.setdefault(rid, []).append(idx)
        idx += 1
    fig.update_layout(**_chart_layout(
        title=dict(text="Training Loss", font=dict(size=14)),
        xaxis_title="Step", yaxis_title="Loss (nats)",
    ))
    return fig.to_json(), trace_map


def build_val_loss_fig(runs: list[dict], colors: dict[str, str]) -> tuple[str, dict]:
    fig = go.Figure()
    trace_map: dict[str, list[int]] = {}
    idx = 0
    for run in runs:
        rid = run["meta"]["run_id"]
        color = colors[rid]
        steps = [e["step"] for e in run["val_steps"]]
        losses = [e["val_loss"] for e in run["val_steps"]]
        if not steps:
            continue
        fig.add_trace(go.Scattergl(
            x=steps, y=losses, name=rid,
            line=dict(color=color, width=2.5),
            mode="lines+markers", marker=dict(size=5, symbol="circle"),
            hovertemplate="<b>%{fullData.name}</b><br>step %{x:,}<br>val_loss %{y:.4f}<extra></extra>",
        ))
        trace_map.setdefault(rid, []).append(idx)
        idx += 1
    fig.update_layout(**_chart_layout(
        title=dict(text="Validation Loss", font=dict(size=14)),
        xaxis_title="Step", yaxis_title="Val Loss (nats)",
    ))
    return fig.to_json(), trace_map


def build_bpb_fig(runs: list[dict], colors: dict[str, str]) -> tuple[str, dict]:
    fig = go.Figure()
    trace_map: dict[str, list[int]] = {}
    idx = 0
    for run in runs:
        rid = run["meta"]["run_id"]
        color = colors[rid]
        steps = [e["step"] for e in run["val_steps"]]
        bpbs = [e["val_bpb"] for e in run["val_steps"]]
        if not steps:
            continue
        fig.add_trace(go.Scattergl(
            x=steps, y=bpbs, name=rid,
            line=dict(color=color, width=2.5),
            mode="lines+markers", marker=dict(size=5, symbol="circle"),
            hovertemplate="<b>%{fullData.name}</b><br>step %{x:,}<br>BPB %{y:.4f}<extra></extra>",
        ))
        trace_map.setdefault(rid, []).append(idx)
        idx += 1
    fig.update_layout(**_chart_layout(
        title=dict(text="Validation Bits-per-Byte", font=dict(size=14)),
        xaxis_title="Step", yaxis_title="BPB",
    ))
    return fig.to_json(), trace_map


def build_throughput_fig(runs: list[dict], colors: dict[str, str]) -> tuple[str, dict]:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Throughput (tok/s)", "Step Time (ms)"),
        vertical_spacing=0.10, shared_xaxes=True,
    )
    trace_map: dict[str, list[int]] = {}
    idx = 0
    for run in runs:
        rid = run["meta"]["run_id"]
        color = colors[rid]
        steps_t = [e["step"] for e in run["train_steps"] if e.get("tok_s")]
        tok_s = [e["tok_s"] for e in run["train_steps"] if e.get("tok_s")]
        steps_m = [e["step"] for e in run["train_steps"] if e.get("step_time_ms")]
        ms = [e["step_time_ms"] for e in run["train_steps"] if e.get("step_time_ms")]
        indices = []
        if steps_t:
            fig.add_trace(go.Scattergl(
                x=steps_t, y=tok_s, name=rid, legendgroup=rid,
                line=dict(color=color, width=2),
                hovertemplate="<b>%{fullData.name}</b><br>step %{x:,}<br>%{y:,.0f} tok/s<extra></extra>",
            ), row=1, col=1)
            indices.append(idx); idx += 1
        if steps_m:
            fig.add_trace(go.Scattergl(
                x=steps_m, y=ms, name=rid, legendgroup=rid, showlegend=False,
                line=dict(color=color, width=2),
                hovertemplate="<b>%{fullData.name}</b><br>step %{x:,}<br>%{y:.1f} ms<extra></extra>",
            ), row=2, col=1)
            indices.append(idx); idx += 1
        if indices:
            trace_map.setdefault(rid, []).extend(indices)

    fig.update_layout(**_chart_layout(height=640))
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor="rgba(255,255,255,0.04)")})
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="tok/s", row=1, col=1)
    fig.update_yaxes(title_text="ms / step", row=2, col=1)
    return fig.to_json(), trace_map


def build_wallclock_fig(runs: list[dict], colors: dict[str, str]) -> tuple[str, dict]:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Training Loss vs Time", "Validation BPB vs Time"),
        vertical_spacing=0.10, shared_xaxes=True,
    )
    trace_map: dict[str, list[int]] = {}
    idx = 0
    for run in runs:
        rid = run["meta"]["run_id"]
        color = colors[rid]
        indices = []

        t_s = [e["train_time_ms"] / 1000 for e in run["train_steps"] if e.get("train_time_ms") is not None]
        t_l = [e["train_loss"] for e in run["train_steps"] if e.get("train_time_ms") is not None]
        if t_s:
            fig.add_trace(go.Scattergl(
                x=t_s, y=t_l, name=rid, legendgroup=rid,
                line=dict(color=color, width=2),
                hovertemplate="<b>%{fullData.name}</b><br>%{x:.0f}s<br>loss %{y:.4f}<extra></extra>",
            ), row=1, col=1)
            indices.append(idx); idx += 1

        vt_s = [e["train_time_ms"] / 1000 for e in run["val_steps"] if e.get("train_time_ms") is not None]
        vt_b = [e["val_bpb"] for e in run["val_steps"] if e.get("train_time_ms") is not None]
        if vt_s:
            fig.add_trace(go.Scattergl(
                x=vt_s, y=vt_b, name=rid, legendgroup=rid, showlegend=False,
                line=dict(color=color, width=2.5),
                mode="lines+markers", marker=dict(size=5),
                hovertemplate="<b>%{fullData.name}</b><br>%{x:.0f}s<br>BPB %{y:.4f}<extra></extra>",
            ), row=2, col=1)
            indices.append(idx); idx += 1

        if indices:
            trace_map.setdefault(rid, []).extend(indices)

    # 600s budget line
    for row in [1, 2]:
        fig.add_vline(x=600, line_dash="dot", line_color="rgba(224,159,62,0.5)",
                      annotation=dict(text="10 min", font=dict(color=TEXT_MUTED, size=10)),
                      row=row, col=1)

    fig.update_layout(**_chart_layout(height=640))
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor="rgba(255,255,255,0.04)")})
    fig.update_xaxes(title_text="Wall-clock (s)", row=2, col=1)
    fig.update_yaxes(title_text="Train Loss", row=1, col=1)
    fig.update_yaxes(title_text="Val BPB", row=2, col=1)
    return fig.to_json(), trace_map


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html_mod.escape(s, quote=True)


def _fmt_bpb(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _fmt_tok(v: float | None) -> str:
    return f"{v:,.0f}" if v is not None else "—"


def _fmt_ms(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _fmt_time(ms: float | None) -> str:
    if ms is None:
        return "—"
    s = ms / 1000
    if s < 120:
        return f"{s:.0f}s"
    return f"{s / 60:.1f}m"


def _fmt_mb(b: int | None) -> str:
    return f"{b / 1e6:.2f}" if b is not None else "—"


def _delta_html(val: float | None, ref: float | None, lower_better: bool = True) -> str:
    if val is None or ref is None:
        return ""
    d = val - ref
    if abs(d) < 1e-6:
        return '<span class="delta-neutral">±0</span>'
    good = (d < 0) if lower_better else (d > 0)
    cls = "delta-good" if good else "delta-bad"
    sign = "+" if d > 0 else ""
    return f'<span class="{cls}">{sign}{d:.4f}</span>'


def _delta_pct_html(val: float | None, ref: float | None, higher_better: bool = True) -> str:
    if val is None or ref is None or ref == 0:
        return ""
    pct = 100 * (val - ref) / abs(ref)
    if abs(pct) < 0.05:
        return '<span class="delta-neutral">±0%</span>'
    good = (pct > 0) if higher_better else (pct < 0)
    cls = "delta-good" if good else "delta-bad"
    sign = "+" if pct > 0 else ""
    return f'<span class="{cls}">{sign}{pct:.1f}%</span>'


def generate_html(
    runs: list[dict],
    baseline_id: str | None,
    colors: dict[str, str],
    output: Path,
) -> None:
    # Sort: baseline first, then by BPB ascending
    def sort_key(r: dict) -> tuple:
        rid = r["meta"]["run_id"]
        bpb = (r["meta"].get("results") or {}).get("final_int8_bpb")
        if bpb is None:
            bpb = (r["meta"].get("results") or {}).get("final_val_bpb")
        return (0 if rid == baseline_id else 1, bpb or 999)
    runs.sort(key=sort_key)

    # ── Build all charts ─────────────────────────────────────────────
    charts_info: list[tuple[str, str, str, dict]] = []  # (div_id, title, fig_json, trace_map)
    for div_id, title, builder in [
        ("chart-train-loss", "Training Loss", build_train_loss_fig),
        ("chart-val-loss", "Validation Loss", build_val_loss_fig),
        ("chart-bpb", "Validation BPB", build_bpb_fig),
        ("chart-throughput", "Throughput & Timing", build_throughput_fig),
        ("chart-wallclock", "Wall-clock View", build_wallclock_fig),
    ]:
        fig_json, trace_map = builder(runs, colors)
        charts_info.append((div_id, title, fig_json, trace_map))

    # Merge trace maps: {div_id: {run_id: [indices]}}
    all_trace_maps = {ci[0]: ci[3] for ci in charts_info}

    # ── KPI values ───────────────────────────────────────────────────
    bl_bpb = None
    best_bpb = None
    best_id = None
    for r in runs:
        res = r["meta"].get("results") or {}
        bpb = res.get("final_int8_bpb") or res.get("final_val_bpb")
        if r["meta"]["run_id"] == baseline_id and bpb is not None:
            bl_bpb = bpb
        if bpb is not None and (best_bpb is None or bpb < best_bpb):
            best_bpb = bpb
            best_id = r["meta"]["run_id"]
    delta_bpb = (best_bpb - bl_bpb) if (best_bpb is not None and bl_bpb is not None) else None

    # ── Run selector chips ───────────────────────────────────────────
    chips_html = []
    for run in runs:
        rid = run["meta"]["run_id"]
        color = colors[rid]
        res = run["meta"].get("results") or {}
        bpb = res.get("final_int8_bpb") or res.get("final_val_bpb")
        bpb_str = f" ({bpb:.4f})" if bpb else ""
        is_bl = rid == baseline_id
        label = rid
        if len(label) > 32:
            label = label[:31] + "\u2026"
        extra_cls = " chip-baseline" if is_bl else ""
        chips_html.append(
            f'<button class="chip active{extra_cls}" data-run="{_esc(rid)}" '
            f'style="--chip-color:{color}" '
            f'title="{_esc(rid)}{bpb_str}">'
            f'<span class="chip-dot"></span>'
            f'<span class="chip-label">{_esc(label)}{bpb_str}</span>'
            f'</button>'
        )

    # ── Comparison table ─────────────────────────────────────────────
    bl_res = {}
    for r in runs:
        if r["meta"]["run_id"] == baseline_id:
            bl_res = r["meta"].get("results") or {}
            break
    bl_tok = bl_res.get("mean_tok_s")
    bl_step = bl_res.get("mean_step_ms")
    bl_bpb_ref = bl_res.get("final_int8_bpb") or bl_res.get("final_val_bpb")

    table_rows = []
    for run in runs:
        m = run["meta"]
        res = m.get("results") or {}
        rid = m["run_id"]
        is_bl = rid == baseline_id
        color = colors[rid]
        bpb = res.get("final_int8_bpb") or res.get("final_val_bpb")
        status = m.get("status", "?")
        status_map = {"completed": ("done", "Completed"), "running": ("pending", "Running"),
                      "failed": ("error", "Failed"), "incomplete": ("warn", "Incomplete")}
        s_cls, s_tip = status_map.get(status, ("unknown", status))
        hyp = m.get("hypothesis", "")

        row_cls = "row-baseline" if is_bl else ""
        table_rows.append(f'''<tr class="{row_cls}" data-run="{_esc(rid)}">
            <td><span class="status-dot status-{s_cls}" title="{_esc(s_tip)}"></span></td>
            <td><span class="color-dot" style="background:{color}"></span> {_esc(rid[:40])}</td>
            <td class="cell-hyp" title="{_esc(hyp)}">{_esc(hyp[:70])}</td>
            <td class="num"><b>{_fmt_bpb(bpb)}</b></td>
            <td class="num">{_delta_html(bpb, bl_bpb_ref)}</td>
            <td class="num">{_fmt_tok(res.get("mean_tok_s"))}</td>
            <td class="num">{_delta_pct_html(res.get("mean_tok_s"), bl_tok, higher_better=True)}</td>
            <td class="num">{_fmt_ms(res.get("mean_step_ms"))}</td>
            <td class="num">{res.get("total_steps", "—")}</td>
            <td class="num">{_fmt_time(res.get("total_train_time_ms"))}</td>
            <td class="num">{_fmt_mb(res.get("artifact_bytes"))}</td>
        </tr>''')

    # ── Hyperparameter diff ──────────────────────────────────────────
    all_params: dict[str, dict[str, object]] = {}
    for run in runs:
        hp = run["meta"].get("hyperparameters", {})
        rid = run["meta"]["run_id"]
        for k, v in hp.items():
            all_params.setdefault(k, {})[rid] = v

    varying: dict[str, dict[str, object]] = {}
    for param, vals in all_params.items():
        unique = set(str(v) for v in vals.values())
        if len(unique) > 1:
            varying[param] = vals

    if varying:
        run_ids = [r["meta"]["run_id"] for r in runs]
        hp_header = "".join(f'<th title="{_esc(rid)}">{_esc(rid[:18])}</th>' for rid in run_ids)
        hp_rows = []
        bl_vals = {p: varying[p].get(baseline_id) for p in varying} if baseline_id else {}
        for param in sorted(varying):
            cells = []
            for rid in run_ids:
                val = varying[param].get(rid, "—")
                is_diff = baseline_id and rid != baseline_id and str(val) != str(bl_vals.get(param))
                cls = ' class="hp-changed"' if is_diff else ""
                cells.append(f"<td{cls}>{val}</td>")
            hp_rows.append(f'<tr><td><code>{_esc(param)}</code></td>{"".join(cells)}</tr>')
        diff_html = f'''<div class="table-wrap"><table class="hp-table">
            <thead><tr><th>Parameter</th>{hp_header}</tr></thead>
            <tbody>{"".join(hp_rows)}</tbody></table></div>'''
    else:
        diff_html = '<p class="muted">All hyperparameters identical across runs.</p>'

    # ── Chart divs ───────────────────────────────────────────────────
    chart_divs = []
    for div_id, title, _, _ in charts_info:
        chart_divs.append(
            f'<div class="chart-section" id="section-{div_id}">'
            f'<div class="chart-container"><div id="{div_id}"></div></div></div>'
        )

    # ── Chart nav tabs ───────────────────────────────────────────────
    nav_items = []
    for div_id, title, _, _ in charts_info:
        nav_items.append(f'<button class="nav-tab" data-target="section-{div_id}">{_esc(title)}</button>')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Assemble HTML ────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parameter Golf Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root {{
    --bg-page: {BG_PAGE};
    --bg-card: {BG_CARD};
    --bg-card-hover: {BG_CARD_HOVER};
    --border: {BORDER};
    --text-1: {TEXT_PRIMARY};
    --text-2: {TEXT_SECONDARY};
    --text-muted: {TEXT_MUTED};
    --accent: {ACCENT};
    --green: {GREEN};
    --red: {RED};
}}
*, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; line-height: 1.5;
    background: var(--bg-page); color: var(--text-1);
    -webkit-font-smoothing: antialiased;
}}

/* ── Layout ────────────────────────────────────── */
.page {{ max-width: 1440px; margin: 0 auto; padding: 24px 32px 80px; }}

/* ── Header ────────────────────────────────────── */
.header {{
    display: flex; justify-content: space-between; align-items: flex-end;
    padding-bottom: 20px; margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
}}
.header h1 {{
    font-size: 20px; font-weight: 700; letter-spacing: -0.02em;
    color: var(--text-1);
}}
.header h1 span {{ color: var(--accent); }}
.header-meta {{ font-size: 11px; color: var(--text-muted); text-align: right; }}

/* ── KPI Cards ─────────────────────────────────── */
.kpi-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 28px;
}}
.kpi {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px;
}}
.kpi-label {{ font-size: 11px; font-weight: 500; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
.kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 600; }}
.kpi-sub {{ font-size: 11px; color: var(--text-2); margin-top: 4px; }}

/* ── Section headers ───────────────────────────── */
.section-title {{
    font-size: 13px; font-weight: 600; color: var(--text-2);
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 28px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}

/* ── Run selector chips ────────────────────────── */
.chip-bar {{
    display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px;
    align-items: center;
}}
.chip-bar-actions {{
    display: flex; gap: 6px; margin-right: 12px;
}}
.chip-bar-actions button {{
    font-family: inherit; font-size: 11px; font-weight: 500;
    background: transparent; color: var(--text-muted); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; cursor: pointer; transition: all 0.15s;
}}
.chip-bar-actions button:hover {{ color: var(--text-1); border-color: var(--text-2); }}
.chip {{
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 20px; padding: 4px 12px 4px 8px;
    cursor: pointer; transition: all 0.15s; user-select: none;
    color: var(--text-2);
}}
.chip:hover {{ border-color: var(--text-2); }}
.chip.active {{ border-color: var(--chip-color); color: var(--text-1); }}
.chip.active .chip-dot {{ opacity: 1; }}
.chip-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--chip-color); opacity: 0.3;
    transition: opacity 0.15s; flex-shrink: 0;
}}
.chip-baseline {{ font-weight: 600; }}

/* ── Table ─────────────────────────────────────── */
.table-wrap {{
    overflow-x: auto; margin-bottom: 8px;
    border: 1px solid var(--border); border-radius: 10px;
    background: var(--bg-card);
}}
table {{
    width: 100%; border-collapse: collapse;
    font-size: 12px;
}}
th {{
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--text-muted);
    padding: 10px 12px; text-align: left;
    background: rgba(255,255,255,0.02); cursor: pointer;
    user-select: none; white-space: nowrap;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0;
}}
th:hover {{ color: var(--text-2); }}
th .sort-arrow {{ font-size: 9px; margin-left: 3px; opacity: 0.4; }}
th.sorted .sort-arrow {{ opacity: 1; color: var(--accent); }}
td {{
    padding: 8px 12px; white-space: nowrap;
    border-top: 1px solid rgba(255,255,255,0.03);
}}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
tr.row-baseline td {{ background: rgba(51,92,103,0.1); }}
tr.row-baseline:hover td {{ background: rgba(51,92,103,0.15); }}
.num {{ font-family: 'JetBrains Mono', monospace; text-align: right; }}
.cell-hyp {{ max-width: 280px; overflow: hidden; text-overflow: ellipsis; color: var(--text-2); }}
.color-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
.status-dot {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
}}
.status-done {{ background: var(--green); }}
.status-pending {{ background: var(--accent); }}
.status-error {{ background: var(--red); }}
.status-warn {{ background: #fbbf24; }}
.status-unknown {{ background: var(--text-muted); }}

.delta-good {{ color: var(--green); font-family: 'JetBrains Mono', monospace; font-size: 11px; }}
.delta-bad {{ color: var(--red); font-family: 'JetBrains Mono', monospace; font-size: 11px; }}
.delta-neutral {{ color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 11px; }}

/* ── Hyperparam diff ───────────────────────────── */
.hp-table td, .hp-table th {{ font-size: 11px; padding: 6px 10px; }}
.hp-table code {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-2); }}
.hp-changed {{ color: var(--accent) !important; font-weight: 600; }}

/* ── Chart nav ─────────────────────────────────── */
.chart-nav {{
    display: flex; gap: 2px; margin-bottom: 0;
    background: var(--bg-card); border: 1px solid var(--border);
    border-bottom: none; border-radius: 10px 10px 0 0;
    padding: 4px 4px 0; overflow-x: auto;
}}
.nav-tab {{
    font-family: inherit; font-size: 12px; font-weight: 500;
    background: transparent; color: var(--text-muted);
    border: none; border-bottom: 2px solid transparent;
    padding: 10px 18px; cursor: pointer; transition: all 0.15s;
    white-space: nowrap;
}}
.nav-tab:hover {{ color: var(--text-2); }}
.nav-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

/* ── Chart containers ──────────────────────────── */
.charts-wrapper {{
    border: 1px solid var(--border); border-top: none;
    border-radius: 0 0 10px 10px; background: var(--bg-card);
    padding: 16px; overflow: hidden;
}}
.chart-section {{ display: none; }}
.chart-section.active {{ display: block; }}
.chart-container {{ width: 100%; min-height: 520px; }}

.muted {{ color: var(--text-muted); font-style: italic; padding: 12px 0; }}
</style>
</head>
<body>
<div class="page">

<!-- Header -->
<div class="header">
    <div>
        <h1>Parameter <span>Golf</span></h1>
    </div>
    <div class="header-meta">
        {len(runs)} runs &middot; generated {now}
    </div>
</div>

<!-- KPI Cards -->
<div class="kpi-row">
    <div class="kpi">
        <div class="kpi-label">Baseline BPB</div>
        <div class="kpi-value">{f"{bl_bpb:.4f}" if bl_bpb else "—"}</div>
        <div class="kpi-sub">{_esc(baseline_id or "none")} (int8)</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Best BPB</div>
        <div class="kpi-value" style="color:var(--green)">{f"{best_bpb:.4f}" if best_bpb else "—"}</div>
        <div class="kpi-sub">{_esc(best_id or "—")}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">&Delta; vs Baseline</div>
        <div class="kpi-value" style="color:{'var(--green)' if delta_bpb and delta_bpb < 0 else 'var(--red)' if delta_bpb and delta_bpb > 0 else 'var(--text-muted)'}">{f"{delta_bpb:+.4f}" if delta_bpb else "—"}</div>
        <div class="kpi-sub">{"improvement" if delta_bpb and delta_bpb < 0 else "regression" if delta_bpb and delta_bpb > 0 else ""}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Total Runs</div>
        <div class="kpi-value">{len(runs)}</div>
        <div class="kpi-sub">discovered from outputs/</div>
    </div>
</div>

<!-- Run Selector -->
<div class="section-title">Run Selector</div>
<div class="chip-bar">
    <div class="chip-bar-actions">
        <button onclick="toggleAll(true)">All</button>
        <button onclick="toggleAll(false)">None</button>
    </div>
    {"".join(chips_html)}
</div>

<!-- Comparison Table -->
<div class="section-title">Run Comparison</div>
<div class="table-wrap">
<table id="comparison-table">
<thead><tr>
    <th></th>
    <th data-sort="str">Run ID <span class="sort-arrow">&udarr;</span></th>
    <th data-sort="str">Hypothesis <span class="sort-arrow">&udarr;</span></th>
    <th data-sort="num" class="num">BPB <span class="sort-arrow">&udarr;</span></th>
    <th class="num">&Delta; BPB</th>
    <th data-sort="num" class="num">tok/s <span class="sort-arrow">&udarr;</span></th>
    <th class="num">&Delta; tok/s</th>
    <th data-sort="num" class="num">ms/step <span class="sort-arrow">&udarr;</span></th>
    <th data-sort="num" class="num">Steps <span class="sort-arrow">&udarr;</span></th>
    <th data-sort="str" class="num">Time <span class="sort-arrow">&udarr;</span></th>
    <th data-sort="num" class="num">Size MB <span class="sort-arrow">&udarr;</span></th>
</tr></thead>
<tbody>{"".join(table_rows)}</tbody>
</table>
</div>

<!-- Charts -->
<div class="section-title">Training Curves</div>
<div class="chart-nav">
    {"".join(nav_items)}
</div>
<div class="charts-wrapper">
    {"".join(chart_divs)}
</div>

<!-- Hyperparameter Diff -->
<div class="section-title">Hyperparameter Diff</div>
{diff_html}

</div><!-- .page -->

<script>
// ── Chart data ──────────────────────────────────
const CHARTS = {json.dumps({ci[0]: ci[2] for ci in charts_info})};
const TRACE_MAPS = {json.dumps(all_trace_maps)};
const CHART_IDS = {json.dumps([ci[0] for ci in charts_info])};
const PLOTLY_CONFIG = {{ responsive: true, displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false }};

// ── Render charts ───────────────────────────────
CHART_IDS.forEach(id => {{
    const fig = JSON.parse(CHARTS[id]);
    Plotly.newPlot(id, fig.data, fig.layout, PLOTLY_CONFIG);
}});

// Show first chart section
document.querySelector('.chart-section').classList.add('active');
document.querySelector('.nav-tab').classList.add('active');

// ── Chart navigation ────────────────────────────
document.querySelectorAll('.nav-tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.chart-section').forEach(s => s.classList.remove('active'));
        tab.classList.add('active');
        const target = document.getElementById(tab.dataset.target);
        target.classList.add('active');
        // Resize the chart now that it is visible
        const chartDiv = target.querySelector('[id^="chart-"]');
        if (chartDiv) Plotly.Plots.resize(chartDiv);
    }});
}});

// ── Run visibility toggle ───────────────────────
function setRunVisible(runId, visible) {{
    CHART_IDS.forEach(chartId => {{
        const map = TRACE_MAPS[chartId] || {{}};
        const indices = map[runId];
        if (indices && indices.length) {{
            Plotly.restyle(chartId, {{ visible: visible }}, indices);
        }}
    }});
    // Update chip state
    const chip = document.querySelector(`.chip[data-run="${{CSS.escape(runId)}}"]`);
    if (chip) chip.classList.toggle('active', visible);
    // Update table row opacity
    const row = document.querySelector(`tr[data-run="${{CSS.escape(runId)}}"]`);
    if (row) row.style.opacity = visible ? '1' : '0.3';
}}

document.querySelectorAll('.chip').forEach(chip => {{
    chip.addEventListener('click', () => {{
        const rid = chip.dataset.run;
        const isActive = chip.classList.contains('active');
        setRunVisible(rid, !isActive);
    }});
}});

function toggleAll(show) {{
    document.querySelectorAll('.chip').forEach(chip => {{
        setRunVisible(chip.dataset.run, show);
    }});
}}

// ── Table sorting ───────────────────────────────
document.querySelectorAll('#comparison-table th[data-sort]').forEach((th, colIdx) => {{
    // Actual column index (skip status column if needed)
    const actualIdx = Array.from(th.parentNode.children).indexOf(th);
    let ascending = true;
    th.addEventListener('click', () => {{
        const tbody = document.querySelector('#comparison-table tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const sortType = th.dataset.sort;
        rows.sort((a, b) => {{
            let va = a.children[actualIdx]?.textContent.trim() || '';
            let vb = b.children[actualIdx]?.textContent.trim() || '';
            if (sortType === 'num') {{
                va = parseFloat(va.replace(/[^0-9.\-]/g, '')) || 0;
                vb = parseFloat(vb.replace(/[^0-9.\-]/g, '')) || 0;
                return ascending ? va - vb : vb - va;
            }}
            return ascending ? va.localeCompare(vb) : vb.localeCompare(va);
        }});
        rows.forEach(r => tbody.appendChild(r));
        // Update sort indicators
        document.querySelectorAll('#comparison-table th').forEach(h => h.classList.remove('sorted'));
        th.classList.add('sorted');
        ascending = !ascending;
    }});
}});

// ── Window resize ───────────────────────────────
window.addEventListener('resize', () => {{
    const activeChart = document.querySelector('.chart-section.active [id^="chart-"]');
    if (activeChart) Plotly.Plots.resize(activeChart);
}});
</script>
</body>
</html>'''

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Dashboard written to {output}")
    print(f"  {len(runs)} runs, {len(charts_info)} charts")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter Golf dashboard generator")
    parser.add_argument("--runs", nargs="*", help="Specific run IDs to include")
    parser.add_argument("--baseline", help="Run ID to use as baseline reference")
    parser.add_argument("-o", "--output", default="outputs/dashboard.html",
                        help="Output HTML path (default: outputs/dashboard.html)")
    parser.add_argument("--scan-dirs", nargs="*",
                        help="Extra directories to scan for runs")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    scan_roots = [
        project_dir / "outputs" / "runs",
        project_dir / "outputs" / "baseline",
        project_dir / "logs" / "baseline",
    ]
    if args.scan_dirs:
        scan_roots.extend(Path(d) for d in args.scan_dirs)

    run_dirs = discover_runs(scan_roots)
    if not run_dirs:
        print("No runs found. Scanned:", file=sys.stderr)
        for r in scan_roots:
            print(f"  {r}", file=sys.stderr)
        sys.exit(1)

    all_runs: list[dict] = []
    for d in run_dirs:
        run = load_run(d)
        if run:
            all_runs.append(run)

    if args.runs:
        selected = set(args.runs)
        all_runs = [r for r in all_runs if r["meta"]["run_id"] in selected]

    if not all_runs:
        print("No valid runs found after filtering.", file=sys.stderr)
        sys.exit(1)

    baseline_id = args.baseline
    if not baseline_id:
        for r in all_runs:
            if "baseline" in r["meta"]["run_id"].lower():
                baseline_id = r["meta"]["run_id"]
                break

    colors = assign_colors(all_runs, baseline_id)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path

    generate_html(all_runs, baseline_id, colors, output_path)


if __name__ == "__main__":
    main()
