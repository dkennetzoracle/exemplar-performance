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
"""Join the GB300 and VR200 CSVs into one side-by-side table per workload.

The two machines are matched on (container, execution path, MBS, GBS). Two
things make a naive join on the CSV's own columns wrong:

* ``precision`` reports the *preset*, not what executed. llama31_8b has no
  bf16 preset, so a bf16-fallback run selects the nvfp4 preset and overrides
  the precision -- the row reads "nvfp4" on a bf16 run. Joining on it would
  pair a bf16 row with a genuine nvfp4 one.
* recompute is not a column at all, yet on VR200 it is the difference between
  6,543 and 9,210 tokens/s/GPU at otherwise identical shape.

So each row is reduced to an explicit ``path`` label that states what actually
ran, and that label is part of the key. Workloads are emitted as separate
tables because their fixed parameters differ (sequence length, expert
parallelism, MoE dispatcher) and the numbers are not comparable across them.

Usage:
    ./make_comparison.py [-g GB300_CSV] [-v VR200_CSV] [-o out.md]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Fixed parameters per workload, for the table captions. These differ between
# the two models, which is why they get separate tables.
WORKLOAD_META = {
    "llama3.1-8b": dict(
        title="llama3.1 8B (dense)",
        params="seq_len 8192 · TP=1 PP=1 CP=1 · DP=4 · 4 GPUs",
    ),
    "qwen3-30b-a3b": dict(
        title="Qwen3 30B-A3B (MoE)",
        params="seq_len 4096 · TP=1 PP=1 CP=1 · **EP=4** · DP=4 · "
               "MoE dispatcher **alltoall** · 4 GPUs",
    ),
}


def path_label(row: dict) -> str:
    """What actually executed, not what the preset was called."""
    tag = row["tag"]
    fallback = row.get("is_fallback", "").strip().lower() in ("true", "1")
    if not fallback:
        # The committed GB300 rows predate the is_fallback column.
        fallback = "fallback" in tag

    if not fallback:
        prec = row["precision"] or "?"
        # config_variant is a distinct preset, not a duplicate: on GB300,
        # nvfp4 v1 and v2 at MBS=4 GBS=16 differ (56,988 vs 56,594 tok/s/GPU).
        # Without this the two collide on the key and one is silently dropped.
        if (row.get("config_variant") or "v1") != "v1":
            prec += f" {row['config_variant']}"
        return prec

    # Recompute is only meaningful for the dense fallback: qwen3's preset uses
    # cuda_graph_impl=transformer_engine, which asserts against full recompute,
    # so it never has it.
    if row["workload"] == "qwen3-30b-a3b":
        return "bf16 (fallback)"
    if "norecompute" in tag:
        return "bf16 (fallback, no recompute)"
    if "recompute" in tag:
        return "bf16 (fallback, recompute)"
    # GB300's fallback rows never varied recompute -- that is the open pending
    # item on that side -- so they are the recompute-on config by construction.
    return "bf16 (fallback, recompute)"


def load(path: str) -> dict:
    if not os.path.exists(path):
        print(f"warning: {path} not found; that side will be blank", file=sys.stderr)
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        if r["tag"].startswith("VR200-reference-"):
            continue  # synthetic row; the measured ref- row covers it
        key = (r["workload"], r["container"], path_label(r), r["mbs"], r["gbs"])
        # Prefer a successful row if the same config was run more than once
        # (e.g. an anchor repeat), otherwise keep the failure so it is visible.
        if key not in out or (not out[key]["tokens_s_per_gpu"] and r["tokens_s_per_gpu"]):
            out[key] = r
    return out


def cells(r: dict | None) -> tuple[str, str, str]:
    if r is None:
        return "—", "—", "—"
    if not r["tokens_s_per_gpu"]:
        code = r.get("failure_code") or "failed"
        return f"*{code}*", "", ""
    return r["s_iter"], r["tflops_per_gpu"], f"{int(r['tokens_s_per_gpu']):,}"


def pct(vr: dict | None, gb: dict | None) -> str:
    """VR200 relative to GB300 on tokens/s/GPU. Positive = VR200 faster."""
    if not (vr and gb and vr["tokens_s_per_gpu"] and gb["tokens_s_per_gpu"]):
        return "—"
    v, g = float(vr["tokens_s_per_gpu"]), float(gb["tokens_s_per_gpu"])
    return f"{(v / g - 1) * 100:+.1f}%"


# Order paths so the comparable bf16 rows lead and quantized follows.
PATH_ORDER = {
    "bf16 (fallback, recompute)": 0,
    "bf16 (fallback, no recompute)": 1,
    "bf16 (fallback)": 2,
    "bf16": 3,
    "fp8_cs": 4,
    "fp8_mx": 5,
    "nvfp4": 6,
}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gb300", default=os.path.join(here, "perf_matrix.csv"))
    ap.add_argument("-v", "--vr200", default=os.environ.get(
        "VR200_CSV", os.path.join(os.environ.get("LLMB_INSTALL", "/mnt/localdisk/llmb"),
                                  "perf-matrix-results", "perf_matrix.csv")))
    ap.add_argument("-o", "--out", default=os.path.join(here, "COMPARISON.md"))
    args = ap.parse_args()

    gb, vr = load(args.gb300), load(args.vr200)
    keys = sorted(set(gb) | set(vr),
                  key=lambda k: (k[0], k[1], PATH_ORDER.get(k[2], 9), k[2],
                                 int(k[3] or 0), int(k[4] or 0)))

    lines = [
        "# GB300 (sm_103) vs Vera Rubin (sm_107) — raw results",
        "",
        "Generated by `make_comparison.py`; regenerate after any new run.",
        "Both machines: 4 GPUs, Megatron-Bridge `b50da4c`, 50 steps, metrics",
        "averaged over iterations 35–44 by the recipe's own parser.",
        "",
        "**Reading the table.** `precision` states what actually *executed*, not",
        "the preset name — a bf16 fallback run reports `nvfp4` in the raw CSV",
        "because llama31_8b has no bf16 preset, so the nvfp4 preset is selected",
        "and the precision overridden. Recompute is part of the label because at",
        "identical batch shape it is worth 40% on VR200. `%diff` is VR200 versus",
        "GB300 on tokens/s/GPU; positive means VR200 is faster. Italic cells are",
        "failures, showing the cause.",
        "",
        "**Which metric to trust.** `tok/s/GPU` = GBS × seq_len ÷ s/iter ÷ GPUs.",
        "Prefer it: `s/iter` scales with GBS so it cannot compare different batch",
        "shapes, and TFLOPS/GPU comes from the framework's own model-FLOPs",
        "accounting, which changed by a factor 0.953 between the two containers —",
        "so TFLOPS is comparable within a container but not across them.",
        "",
    ]

    for wl in ("llama3.1-8b", "qwen3-30b-a3b"):
        meta = WORKLOAD_META[wl]
        wl_keys = [k for k in keys if k[0] == wl]
        if not wl_keys:
            continue
        lines += [
            f"## {meta['title']}",
            "",
            meta["params"],
            "",
            "| container | precision | MBS | GBS | GB300 s/iter | GB300 TFLOPS/GPU "
            "| GB300 tok/s/GPU | VR200 s/iter | VR200 TFLOPS/GPU | VR200 tok/s/GPU | %diff |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k in wl_keys:
            g, v = gb.get(k), vr.get(k)
            gs, gt, gk = cells(g)
            vs, vt, vk = cells(v)
            lines.append(
                f"| {k[1]} | {k[2]} | {k[3]} | {k[4]} | {gs} | {gt} | {gk} "
                f"| {vs} | {vt} | {vk} | {pct(v, g)} |")
        lines.append("")

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(keys)} configs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
