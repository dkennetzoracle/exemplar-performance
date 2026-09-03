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
"""Refresh the committed VR200 snapshot from this node's live results.

`vr200_perf_matrix.csv` is the VR200 half of COMPARISON.md, committed so the
table can be regenerated from either machine. It has to stay in step with the
runs as they land, and hand-transcribing it is how the two drift apart --
especially for failure rows, where the interesting content is the attributed
cause rather than a number.

Reads the full 31-column CSV that collect_results.py writes into RESULTS_DIR,
projects it onto the snapshot's 12-column schema, and renames tags to the
snapshot's convention:

    vr-<container digits>-<workload>-<path>-mbs<N>-gbs<N>

Rows already in the snapshot that this node has no log for are preserved, so
running it on a node holding only part of the sweep cannot delete the rest.

Usage:
    ./update_vr200_snapshot.py [-i live.csv] [-o vr200_perf_matrix.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

FIELDS = ["tag", "workload", "container", "precision", "config_variant",
          "mbs", "gbs", "s_iter", "tflops_per_gpu", "tokens_s_per_gpu",
          "failure_code", "is_fallback"]


def path_token(row: dict) -> str:
    """The `<path>` element of the tag: what execution path this row took."""
    tag, wl = row["tag"], row["workload"]
    fallback = str(row.get("is_fallback", "")).strip().lower() in ("true", "1")
    if fallback:
        # Recompute is only a variable for the dense fallback; qwen3's preset
        # uses cuda_graph_impl=transformer_engine, which asserts against full
        # recompute, so it never has it and the token stays plain "fallback".
        if wl == "qwen3-30b-a3b":
            return "fallback"
        if "norecompute" in tag:
            return "norecompute"
        return "recompute"
    # Native rows: the precision is the path. fp8_cs -> fp8cs to keep the tag
    # delimiter unambiguous.
    prec = (row.get("precision") or "na").replace("_", "")
    variant = row.get("config_variant") or "v1"
    return prec if variant == "v1" else f"{prec}-{variant}"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--live", default=os.environ.get(
        "VR200_LIVE_CSV", os.path.join(
            os.environ.get("LLMB_INSTALL", "/mnt/localdisk/llmb"),
            "perf-matrix-results", "perf_matrix.csv")))
    ap.add_argument("-o", "--out", default=os.path.join(here, "vr200_perf_matrix.csv"))
    args = ap.parse_args()

    if not os.path.exists(args.live):
        print(f"error: live CSV not found: {args.live}", file=sys.stderr)
        return 1

    existing = {}
    if os.path.exists(args.out):
        existing = {r["tag"]: r for r in csv.DictReader(open(args.out))}

    fresh = {}
    for r in csv.DictReader(open(args.live)):
        if r["tag"].startswith("VR200-reference-"):
            continue  # synthetic; the measured ref- row covers the same config
        if r.get("failure_code") == "in_flight":
            continue  # still running, or a stale log from a dead node
        cont = r["container"].replace(".", "")
        tag = (f"vr-{cont}-{r['workload']}-{path_token(r)}"
               f"-mbs{r['mbs']}-gbs{r['gbs']}")
        fresh[tag] = {
            "tag": tag,
            "workload": r["workload"],
            "container": r["container"],
            "precision": r["precision"],
            "config_variant": r.get("config_variant") or "v1",
            "mbs": r["mbs"], "gbs": r["gbs"],
            "s_iter": r["s_iter"],
            "tflops_per_gpu": r["tflops_per_gpu"],
            "tokens_s_per_gpu": r["tokens_s_per_gpu"],
            "failure_code": r.get("failure_code", ""),
            "is_fallback": r.get("is_fallback", ""),
        }

    merged = dict(existing)
    merged.update(fresh)  # this node's live data wins for tags it has

    def sort_key(t: str) -> tuple:
        r = merged[t]
        return (r["workload"], r["container"], t)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for t in sorted(merged, key=sort_key):
            w.writerow(merged[t])

    added = len(set(fresh) - set(existing))
    print(f"wrote {args.out}: {len(merged)} rows "
          f"({added} new, {len(existing)} pre-existing, {len(fresh)} refreshed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
