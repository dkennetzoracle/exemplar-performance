#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""Flat CSV + one chart per (workload, container) from the two source CSVs.

`comparison.csv` is the flat form of COMPARISON.md -- one row per config with
both machines side by side -- for loading into a spreadsheet. It doubles as the
table view the charts need for accessibility, since a PNG carries no hover
layer.

Chart form is a **grouped** bar, not stacked. The two machines' throughputs
are not parts of a whole, so stacking them would draw a total that does not
exist; `choosing-a-form.md` lists stacked bar for part-to-whole and
grouped/stacked for "tell distinct series apart", and only the grouped reading
is true here.

The y-axis is linear and shared across panels, and starts at zero. Only 8 of
the 65 configs ran on both machines, so most bars are single-series -- the
quantized paths are GB300-only because VR200 has no ptxas target for sm_107.
That puts bf16 (~5-10k tok/s/GPU) and nvfp4 (~60k) on one scale, which
compresses the bf16 end; it is left that way on purpose. The 6x gap between
what GB300 reaches with quantization and the best VR200 bf16 number is the
headline, a broken or log axis on bars would misstate exactly that ratio, and
every bar carries its value as a direct label.

Container is deliberately not in the bar label -- it is the panel title
instead, one panel per container on a shared y-axis -- so the labels stay
short and `bf16_fbre_1_8` under 26.06.01 cannot be confused with the same
config under 26.08.00. Label scheme:

    <precision>_<path>_<mbs>_<gbs>        e.g. bf16_fbre_1_8

    fbre  bf16 fallback, full recompute      fbnr  bf16 fallback, no recompute
    fb    bf16 fallback (MoE: no recompute   nat   native, no arch workarounds
          variable -- its preset forbids it)

Colours are reference-palette categorical slots 1 and 2, used unmodified.
Those two are documented as validated all-pairs in both modes and both clear
3:1 on the light surface. Re-run scripts/validate_palette.js if you change
them.

Usage:
    ./make_plots.py [-g GB300_CSV] [-v VR200_CSV] [-d OUTDIR]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_comparison import PATH_ORDER, WORKLOAD_META, load  # noqa: E402

SERIES_1 = "#2a78d6"   # blue  - GB300
SERIES_2 = "#eb6834"   # orange - VR200
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8981"
GRID = "#e6e5e0"

PATH_SHORT = {
    "bf16 (fallback, recompute)": "fbre",
    "bf16 (fallback, no recompute)": "fbnr",
    "bf16 (fallback)": "fb",
}
# Abbreviations for the failure codes, so a missing bar still says why.
CAUSE_SHORT = {
    "oom_dense": "OOM", "oom_moe": "OOM",
    "ptxas_no_target": "no ptxas tgt",
    "no_cubin_for_arch": "no cubin",
    "fp8_dpa_no_backend": "fp8 DPA",
    "preset_is_8gpu_shape": "8-GPU preset",
    "cuda_graph_assert": "cudagraph",
    "unknown": "no data", "in_flight": "no data", "metrics_unparsed": "no data",
}


def short_label(key: tuple) -> str:
    """(workload, container, path, mbs, gbs) -> bf16_fbre_1_8"""
    _, _, path, mbs, gbs = key
    if path in PATH_SHORT:
        prec, tag = "bf16", PATH_SHORT[path]
    else:
        prec, tag = path.replace("_", "").replace(" ", "-"), "nat"
    return f"{prec}_{tag}_{mbs}_{gbs}"


def tok(row: dict | None) -> float | None:
    if not row or not row.get("tokens_s_per_gpu"):
        return None
    return float(row["tokens_s_per_gpu"])


def write_csv(gb: dict, vr: dict, keys: list, out: str) -> None:
    cols = ["workload", "container", "config", "precision", "mbs", "gbs",
            "gb300_s_iter", "gb300_tflops_per_gpu", "gb300_tokens_s_per_gpu",
            "gb300_status",
            "vr200_s_iter", "vr200_tflops_per_gpu", "vr200_tokens_s_per_gpu",
            "vr200_status", "vr200_vs_gb300_pct"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in keys:
            g, v = gb.get(k), vr.get(k)
            tg, tv = tok(g), tok(v)
            row = dict(workload=k[0], container=k[1], config=short_label(k),
                       precision=k[2], mbs=k[3], gbs=k[4])
            for pre, r, t in (("gb300", g, tg), ("vr200", v, tv)):
                row[f"{pre}_s_iter"] = (r or {}).get("s_iter", "")
                row[f"{pre}_tflops_per_gpu"] = (r or {}).get("tflops_per_gpu", "")
                row[f"{pre}_tokens_s_per_gpu"] = f"{t:.0f}" if t else ""
                row[f"{pre}_status"] = (
                    "" if r is None else ("ok" if t else (r.get("failure_code") or "failed")))
            row["vr200_vs_gb300_pct"] = f"{(tv / tg - 1) * 100:.1f}" if (tg and tv) else ""
            w.writerow(row)
    print(f"wrote {out} ({len(keys)} configs)")


def _panel(ax, keys, gb, vr, top, container, show_ylabel):
    labels = [short_label(k) for k in keys]
    gvals = [tok(gb.get(k)) or 0 for k in keys]
    vvals = [tok(vr.get(k)) or 0 for k in keys]
    n = len(keys)

    ax.set_facecolor(SURFACE)
    x = range(n)
    w = 0.40
    gap = 0.015  # surface gap between the two adjacent bars in a group
    b1 = ax.bar([i - w / 2 - gap for i in x], gvals, w, label="GB300 (sm_103)",
                color=SERIES_1, linewidth=0)
    b2 = ax.bar([i + w / 2 + gap for i in x], vvals, w, label="VR200 (sm_107)",
                color=SERIES_2, linewidth=0)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=INK_2, labelsize=8, length=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.2, color=INK_2)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, top * 1.10)
    ax.set_title(f"nemo:{container}", color=INK_2, fontsize=9, loc="left", pad=9)
    if show_ylabel:
        ax.set_ylabel("tokens / s / GPU", color=INK_2, fontsize=9)
    else:
        ax.tick_params(labelleft=False)

    # Direct labels on every bar: 2 series, and with a shared linear axis the
    # bf16 end is too short to read off the grid. A failed run gets its cause
    # in muted italic where the bar would be, so an absent bar is never
    # mistaken for missing data.
    for bars, vals, src in ((b1, gvals, gb), (b2, vvals, vr)):
        for bar, val, k in zip(bars, vals, keys):
            cx = bar.get_x() + bar.get_width() / 2
            if val:
                ax.text(cx, val + top * 0.012, f"{val:,.0f}", ha="center",
                        va="bottom", fontsize=6.4, color=INK, rotation=90)
            else:
                r = src.get(k)
                cause = (CAUSE_SHORT.get(r.get("failure_code", ""), "failed")
                         if r else "not run")
                ax.text(cx, top * 0.012, cause, ha="center", va="bottom",
                        fontsize=6.0, color=INK_MUTED, rotation=90, style="italic")
    return b1, b2


def plot(wl, keys, gb, vr, out):
    """One figure per workload; one panel per container, shared y-axis."""
    keys = [k for k in keys if k[0] == wl]
    # Keep a config only if a machine produced a number. A row that failed on
    # both has no magnitude to plot and lives in comparison.csv instead.
    keys = [k for k in keys if tok(gb.get(k)) or tok(vr.get(k))]
    if not keys:
        return False

    containers = sorted({k[1] for k in keys})
    per = {c: [k for k in keys if k[1] == c] for c in containers}
    top = max(max(tok(gb.get(k)) or 0, tok(vr.get(k)) or 0) for k in keys)

    widths = [max(2.2, 0.60 * len(per[c])) for c in containers]
    fig, axes = plt.subplots(
        1, len(containers), figsize=(sum(widths) + 1.6, 6.1), dpi=200,
        gridspec_kw=dict(width_ratios=widths, wspace=0.06))
    if len(containers) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    handles = None
    for i, (ax, c) in enumerate(zip(axes, containers)):
        b1, b2 = _panel(ax, per[c], gb, vr, top, c, show_ylabel=(i == 0))
        handles = (b1, b2)

    meta = WORKLOAD_META[wl]
    fig.suptitle(f"{meta['title']} — 4 GPUs, tokens/s/GPU", color=INK,
                 fontsize=13, x=0.008, ha="left", y=0.985)
    sub = meta["params"].replace("**", "")
    fig.text(0.008, 0.925, f"{sub}  ·  higher is better", color=INK_2,
             fontsize=8.2, ha="left")

    # Bottom-centre: the title row is the widest element and the tall
    # quantized bars own the top-right, so a top legend collides on one
    # workload or the other depending on figure width.
    leg = fig.legend(handles=list(handles), frameon=False, fontsize=9,
                     loc="lower center", bbox_to_anchor=(0.5, -0.015), ncols=2)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # subplots_adjust, not tight_layout: the panels carry an explicit wspace,
    # which tight_layout overrides and then warns about.
    fig.subplots_adjust(left=0.075, right=0.995, top=0.855, bottom=0.21)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({len(keys)} configs, {len(containers)} panels)")
    return True


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gb300", default=os.path.join(here, "perf_matrix.csv"))
    ap.add_argument("-v", "--vr200", default=os.path.join(here, "vr200_perf_matrix.csv"))
    ap.add_argument("-d", "--outdir", default=os.path.join(here, "plots"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gb, vr = load(args.gb300), load(args.vr200)
    keys = sorted(set(gb) | set(vr),
                  key=lambda k: (k[0], k[1], PATH_ORDER.get(k[2], 9), k[2],
                                 int(k[3] or 0), int(k[4] or 0)))

    write_csv(gb, vr, keys, os.path.join(here, "comparison.csv"))
    for wl in ("llama3.1-8b", "qwen3-30b-a3b"):
        slug = wl.replace(".", "").replace("-", "_")
        plot(wl, keys, gb, vr, os.path.join(args.outdir, f"{slug}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
