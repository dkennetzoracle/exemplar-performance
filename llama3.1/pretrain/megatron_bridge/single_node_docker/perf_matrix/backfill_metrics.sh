#!/bin/bash
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
#
# Re-parse finished runs whose metrics are missing from their perf-matrix log.
#
# Two ways a run ends up needing this:
#   1. the parser's deps were absent when it ran, so launch_local.sh printed
#      "could not import llmb_run.pretrain_log_parser (No module named 'typer')"
#   2. the run crashed AFTER training (e.g. nemo 26.08.00's CUDA-graph assert in
#      the post-training eval), so the launcher never reached its parse step
#      even though iterations 35-44 are all present and valid
#
# The run directories live on the node's local disk, so this must run ON the
# node that produced them. The parser's own output is appended back into the
# perf-matrix log, which is where collect_results.py reads from.
#
# Usage:
#   ./backfill_metrics.sh

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }
if ! python3 -c 'import typer' 2>/dev/null; then
    echo "error: the repo parser needs typer; run setup_images.sh to build the venv" >&2
    exit 1
fi

[[ -d $RESULTS_DIR ]] || { echo "error: RESULTS_DIR=$RESULTS_DIR does not exist" >&2; exit 1; }

n=0
for log in "$RESULTS_DIR"/*.log; do
    [[ -f $log ]] || continue
    grep -q 'TFLOPS/GPU:' "$log" && continue

    # launch_local.sh prints "Log: <run-dir>/log-<exp>.out" as its last line.
    run=$(grep -m1 '^Log: ' "$log" | sed 's/^Log: //')
    [[ -n $run && -f $run ]] || continue

    out=$(python3 "$SND_DIR/parse_results_local.py" "$run" 2>&1) || true
    if grep -q 'TFLOPS/GPU:' <<<"$out"; then
        printf '\n--- backfilled by parse_results_local.py ---\n%s\n' "$out" >>"$log"
        echo "backfilled $(basename "$log" .log)"
        n=$((n + 1))
    fi
done

echo "backfilled $n run(s)"
