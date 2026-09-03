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
"""Turn a RESULTS_DIR of run_matrix.sh logs into one CSV plus a ranked table.

Metric choice matters here, and two of the three obvious metrics are traps:

* ``TFLOPS/GPU`` is read from the framework's own ``MODEL_TFLOP/s/GPU`` log
  line, not computed. Megatron-Bridge's model-FLOPs accounting changed between
  nemo 26.06.01 and 26.08.00 -- two unrelated configs both showed the newer
  container reporting a factor 0.953 of the older one's FLOPs per iteration.
  So TFLOPS is comparable *within* an image and NOT across images. On the
  bf16-fallback row it is actively misleading: 26.08.00 was 1.7% faster by the
  stopwatch while reporting 3.0% *lower* TFLOPS.
* ``s/iter`` is honest wall clock but scales with global batch size, so it
  cannot compare rows with different GBS.
* ``tokens/s/GPU`` = GBS * seq_len / s_iter / gpus is immune to both. It is the
  metric to quote when comparing across images or batch shapes. Sanity check:
  on the one row where all three are valid it reproduces the TFLOPS-derived
  delta exactly.

Usage:
    ./collect_results.py [RESULTS_DIR] [-o out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from glob import glob

# Vera Rubin (compute capability 10.7) reference, recorded on
# nvcr.io/nvidia/nemo:26.06.01 with Megatron-Bridge b50da4c, 4 GPUs, from the
# "Verified state" sections of ../README.md and qwen3's single_node_docker
# README. bf16 fallback is that machine's ceiling, not a handicap: TE ships no
# sm_107a cubin and no PTX, so nothing quantized runs at any setting.
VR200_REFERENCE = {
    "llama3.1-8b": dict(s_iter=2.504, tflops=336.82, gbs=8, seq_len=8192),
    "qwen3-30b-a3b": dict(s_iter=28.135, tflops=214.38, gbs=256, seq_len=4096),
}

# Ordered most-specific first; the first match wins.
CAUSE_PATTERNS = [
    (r"No dot product attention backend is available", "fp8_dpa_no_backend"),
    (r"OutOfMemoryError: CUDA out of memory", "oom"),
    (r"INTERNAL ASSERT FAILED", "cuda_graph_assert"),
    (r"no kernel image is available", "no_cubin_for_arch"),
    (r"is not divisible by TP\*PP\*CP", "preset_is_8gpu_shape"),
    (r"Failed to get config from module_name", "no_such_preset"),
    (r"cuDNN Error.*No valid execution plans", "cudnn_no_attn_plan"),
    (r"Cannot select .*shfl\.sync", "triton_cross_entropy"),
    (r"is not a recognized processor|not defined for option 'gpu-name'", "ptxas_no_target"),
]

CAUSE_TEXT = {
    "fp8_dpa_no_backend": (
        "TransformerEngine has no fp8 attention backend for this arch; every "
        "nvfp4/fp8 preset sets fp8_dot_product_attention=true. Workaround: "
        "mixed_precision.fp8_dot_product_attention=false model.fp8_dot_product_attention=false"),
    "oom_moe": (
        "OOM: the gb300 preset assumes EP=8 across 8 GPUs, so EP=4 on a single "
        "node doubles expert weights per GPU. Lower MBS."),
    "oom_dense": (
        "OOM: activation memory at this MBS exceeds device memory. MBS=4 was "
        "the ceiling for llama3.1 8B on 4 x 284 GB."),
    "cuda_graph_assert": (
        "PyTorch INTERNAL ASSERT in CUDA-graph capture during POST-TRAINING "
        "eval (nemo 26.08.00 / torch 2.13.0a0). Training is unaffected, so "
        "timings from a run that got this far are valid. Workaround: "
        "train.eval_iters=0"),
    "no_cubin_for_arch": (
        "TransformerEngine ships no cubin for this arch and no PTX to JIT from "
        "(te_ptx_entries=0), so the kernel launch fails outright. bf16 only."),
    "preset_is_8gpu_shape": (
        "preset parallelism has TP*PP*CP=8 and cannot shard over 4 GPUs. "
        "Pin TP=1 PP=1 CP=1 (a deviation to report with the number)."),
    "no_such_preset": "no preset exists for this gpu_type/dtype/variant combination.",
    "cudnn_no_attn_plan": "cuDNN built no fused-attention plan for this arch. NVTE_UNFUSED_ATTN=1.",
    "triton_cross_entropy": "TE's fused cross-entropy is Triton. model.cross_entropy_loss_fusion=false",
    "ptxas_no_target": "bundled ptxas has no sm_<cc>a target, so Triton/torch.compile cannot build.",
}


def field(txt: str, label: str) -> str:
    m = re.search(rf"^\s*{re.escape(label)}\s*(.*?)\s*$", txt, re.M)
    return m.group(1).strip() if m else ""


def plan_val(plan: str, key: str) -> str:
    m = re.search(rf"\b{key}=(\S+)", plan)
    return m.group(1) if m else ""


def workload_of(exp: str, tag: str) -> str:
    return "qwen3-30b-a3b" if ("qwen3" in exp or "qwen3" in tag) else "llama3.1-8b"


def parse_log(path: str) -> dict | None:
    tag = os.path.basename(path)[: -len(".log")]
    txt = open(path, errors="replace").read()
    exp = field(txt, "Experiment:")
    if not exp:
        return None  # not a launcher log (or died before the banner)

    plan = field(txt, "Plan:")
    prec_line = field(txt, "Precision:")
    iters = len(re.findall(r"elapsed time per iteration", txt))

    m = re.search(r"s/iter:\s+([\d.]+) \(std ([\d.]+)\)", txt)
    s_iter, s_std = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"TFLOPS/GPU:\s+([\d.]+) \(std ([\d.]+)\)", txt)
    tflops, t_std = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"sequence_length: (\d+)", txt)
    seq_len = int(m.group(1)) if m else None

    wl = workload_of(exp, tag)
    gbs = plan_val(plan, "GBS")
    gpus = re.search(r"GPUs:\s+(\d+)", txt)
    gpus = int(gpus.group(1)) if gpus else 4

    tok_s_gpu = None
    if s_iter and seq_len and gbs.isdigit():
        tok_s_gpu = int(gbs) * seq_len / s_iter / gpus

    # Classify.
    code = ""
    for pat, c in CAUSE_PATTERNS:
        if re.search(pat, txt):
            code = c
            break
    if code == "oom":
        code = "oom_moe" if wl == "qwen3-30b-a3b" else "oom_dense"

    if tflops is not None:
        status = "OK"
        note = ""
        if code == "cuda_graph_assert":
            # Training finished; only the post-training eval blew up.
            status = "OK (eval crashed)"
            note = CAUSE_TEXT[code]
    else:
        status = "FAILED"
        note = CAUSE_TEXT.get(code, "see log")
        if not code:
            code = "unknown" if iters < 50 else "metrics_unparsed"
            if code == "metrics_unparsed":
                note = ("training completed but no metrics; run backfill_metrics.sh "
                        "(the parser needs typer -- see setup_images.sh)")

    return dict(
        tag=tag,
        workload=wl,
        container=field(txt, "Image:").replace("nvcr.io/nvidia/nemo:", ""),
        precision=prec_line.split()[0] if prec_line else "",
        config_variant=(re.search(r"Variant:\s*(\S+)", prec_line).group(1)
                        if "Variant:" in prec_line else ""),
        gpu_type_preset=(field(txt, "GPU type:").split() or [""])[0],
        gpus=gpus,
        tp=plan_val(plan, "TP"), pp=plan_val(plan, "PP"), cp=plan_val(plan, "CP"),
        ep=plan_val(plan, "EP"), dp=plan_val(plan, "DP"),
        mbs=plan_val(plan, "MBS"), gbs=gbs, ga=plan_val(plan, "GA"),
        cuda_graph=plan_val(plan, "CG"), seq_len=seq_len or "",
        iters_completed=iters, status=status,
        s_iter=f"{s_iter:.3f}" if s_iter else "",
        s_iter_std=f"{s_std:.3f}" if s_std is not None else "",
        tflops_per_gpu=f"{tflops:.2f}" if tflops else "",
        tflops_std=f"{t_std:.2f}" if t_std is not None else "",
        tokens_s_per_gpu=f"{tok_s_gpu:.0f}" if tok_s_gpu else "",
        failure_code=code, note=note, experiment=exp,
    )


def vr_row(wl: str) -> dict:
    r = VR200_REFERENCE[wl]
    return dict(
        tag=f"VR200-reference-{wl}", workload=wl, container="26.06.01",
        precision="bf16", config_variant="v1", gpu_type_preset="(cc 10.7)",
        gpus=4, tp="1", pp="1", cp="1", ep="4" if "qwen" in wl else "1", dp="4",
        mbs="1", gbs=str(r["gbs"]), ga="", cuda_graph="", seq_len=r["seq_len"],
        iters_completed=50, status="OK (reference)",
        s_iter=f"{r['s_iter']:.3f}", s_iter_std="",
        tflops_per_gpu=f"{r['tflops']:.2f}", tflops_std="",
        tokens_s_per_gpu=f"{r['gbs'] * r['seq_len'] / r['s_iter'] / 4:.0f}",
        failure_code="", experiment="",
        note="bf16 fallback = this machine's ceiling: TE has no sm_107a cubin and no PTX",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    # Same precedence as lib.sh: RESULTS_DIR, else $LLMB_INSTALL/perf-matrix-results.
    # Deriving from LLMB_INSTALL matters because the runbook tells you to run
    # this as a bare `./collect_results.py`, and the install root is node-local
    # (/mnt/localdisk on the GB300 cluster, /mnt/nvme here).
    default_results = os.environ.get("RESULTS_DIR") or os.path.join(
        os.environ.get("LLMB_INSTALL", "/mnt/localdisk/llmb"), "perf-matrix-results")
    ap.add_argument("results_dir", nargs="?", default=default_results)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    rows = [r for r in (parse_log(p) for p in sorted(glob(f"{args.results_dir}/*.log"))) if r]
    if not rows:
        print(f"no launcher logs found in {args.results_dir}", file=sys.stderr)
        return 1
    rows += [vr_row(w) for w in sorted({r["workload"] for r in rows} & VR200_REFERENCE.keys())]

    out = args.out or os.path.join(args.results_dir, "perf_matrix.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def key(r):
        return -(float(r["tokens_s_per_gpu"]) if r["tokens_s_per_gpu"] else -1)

    for wl in sorted({r["workload"] for r in rows}):
        sub = sorted((r for r in rows if r["workload"] == wl), key=key)
        print(f"\n=== {wl} — ranked by tokens/s/GPU "
              f"(container- and batch-shape-independent) ===")
        print(f"{'tag':<44}{'img':<10}{'prec':<8}{'mbs':>4}{'gbs':>5}"
              f"{'s/iter':>9}{'TFLOPS':>9}{'tok/s/GPU':>11}  status")
        base = next((r for r in sub if r["tag"].startswith("VR200-reference")), None)
        for r in sub:
            vs = ""
            if base and r["tokens_s_per_gpu"] and base["tokens_s_per_gpu"] and r is not base:
                ratio = float(r["tokens_s_per_gpu"]) / float(base["tokens_s_per_gpu"])
                vs = (f"  {ratio:.2f}x VR200" if ratio >= 1
                      else f"  VR200 +{(1 / ratio - 1) * 100:.1f}%")
            print(f"{r['tag']:<44}{r['container']:<10}{r['precision']:<8}"
                  f"{r['mbs']:>4}{r['gbs']:>5}{r['s_iter']:>9}"
                  f"{r['tflops_per_gpu']:>9}{r['tokens_s_per_gpu']:>11}  "
                  f"{r['status']}{vs}")
            if r["failure_code"]:
                print(f"{'':<44}  -> {r['failure_code']}: {r['note'][:96]}")

    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
