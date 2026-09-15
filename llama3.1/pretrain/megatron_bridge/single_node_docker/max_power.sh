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
# DEALINGS IN THE SOFTWARE.
#
# Max-power baseline across all GPUs, via DCGM's targeted_power stress plugin,
# with host-side power sampling.
#
# This is the synthetic ceiling to compare the training and inference power
# traces against -- "what do these GPUs draw when something is actually trying
# to melt them", not "what does a real workload draw".
#
# Why targeted_power and not `diagnostic`. Both were measured on this node for
# 30s at 1s sampling:
#
#   targeted_power   2302 W max   2294 W p95    ~1400 MHz SM
#   diagnostic       2102 W max   1666 W p95    ~2215 MHz SM
#
# targeted_power sits right on the 2300 W cap; diagnostic (the gpu_burn-style
# SM/DGEMM load) runs the clocks 800 MHz higher and still draws 600 W less per
# GPU. So `diagnostic` is the better *compute* stress and the wrong tool for a
# power ceiling.
#
# Why the target is left at its default. `-p targeted_power.target_power=<W>`
# turns the plugin into a pass/fail gate against that number rather than just a
# load, and it reports Fail if the ramp does not get there in time. The default
# target already pins the cap, so setting it explicitly only adds a way to
# fail. Left alone.
#
# The GPUs must be idle. DCGM diag wants them to itself and will refuse or
# misreport if a training or inference job is resident -- so run the three
# power tests sequentially, never overlapped.
#
# Usage:
#   ./max_power.sh                      # 30 min, 30s sampling
#   ./max_power.sh -m 2 -i 15           # 2 min, 15s sampling
#   ./max_power.sh -m 30 -o ~/max.csv
#
# Options:
#   -m MIN    minutes to hold the load (default 30)
#   -i SECS   power sample interval (default 30)
#   -o FILE   output CSV (default ~/maxpower-<timestamp>.csv)

set -eu -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COLLECT=$SCRIPT_DIR/collect_power.sh

MINUTES=30
INTERVAL=30
OUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MINUTES=$2; shift 2 ;;
        -i) INTERVAL=$2; shift 2 ;;
        -o) OUT=$2; shift 2 ;;
        -h|--help) sed -n '23,60p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

OUT=${OUT:-$HOME/maxpower-$(date +%Y%m%d-%H%M%S).csv}
DURATION=$(( MINUTES * 60 ))

# Under ~20s the ramp has not finished and the plugin reports Fail with a low
# "max power did not reach desired power" -- measured at 5s, which only got to
# 470 W. Refuse rather than hand back a bogus Fail.
if [[ $DURATION -lt 20 ]]; then
    echo "error: -m $MINUTES is too short; the power ramp needs ~20s" >&2
    exit 1
fi

command -v dcgmi >/dev/null || { echo "error: dcgmi not found" >&2; exit 1; }
[[ -x $COLLECT ]] || { echo "error: $COLLECT not found" >&2; exit 1; }

if ! pgrep -x nv-hostengine >/dev/null; then
    echo "error: nv-hostengine is not running (try: systemctl start nvidia-dcgm)" >&2
    exit 1
fi

# Anything holding the GPUs invalidates the run, so say so before spending 30
# minutes on it.
BUSY=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
if [[ -n ${BUSY// /} ]]; then
    echo "error: GPUs are in use -- DCGM diag needs them idle. Running now:" >&2
    echo "$BUSY" >&2
    exit 1
fi

# dcgmi's own -t is a whole-diag timeout; without headroom over test_duration
# it kills the plugin mid-hold.
DIAG_TIMEOUT=$(( DURATION + 300 ))

echo "==================================================================="
echo " Max power baseline -- DCGM targeted_power"
echo " Hold:      ${MINUTES} min (${DURATION}s)"
echo " Sampling:  every ${INTERVAL}s -> $OUT"
echo " GPUs:      $(nvidia-smi -L | wc -l), cap $(nvidia-smi --query-gpu=power.limit \
                    --format=csv,noheader,nounits -i 0 | tr -d ' ') W each"
echo "==================================================================="

exec "$COLLECT" -o "$OUT" -i "$INTERVAL" -- \
    dcgmi diag -r targeted_power \
        -p "targeted_power.test_duration=$DURATION" \
        -t "$DIAG_TIMEOUT"
