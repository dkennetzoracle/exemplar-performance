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
# Sample per-GPU power/clock/thermal telemetry to a CSV, on the host, next to a
# training run.
#
# This is deliberately separate from the launcher rather than folded into it.
# ENABLE_GPU_METRICS already exists but routes through NsysPlugin, which only
# samples inside the profile step window (default 45-50), writes into a .nsys-rep
# a teammate then has to open in the GUI, and perturbs the very timings the
# matrix measures. For "watch power across a long run" none of that is wanted:
# the sampling has to cover the whole run, land in something plottable, and cost
# the run nothing. nvidia-smi queries NVML on the host and does neither.
#
# The sampler is also why the run should be long. Power ramps over the first
# several iterations and the clocks settle later still, so a 50-step run is
# mostly transient. See POWER_AND_TUNING.md.
#
# Usage:
#   ./collect_power.sh -o power.csv                 # sample until Ctrl-C
#   ./collect_power.sh -o power.csv -- <command>    # sample while command runs
#   ./collect_power.sh --summarize power.csv        # stats from a finished CSV
#
# Options:
#   -o FILE   output CSV (default: power-<timestamp>.csv)
#   -i SECS   sample interval, default 1. Below ~0.1 NVML starts returning
#             repeats rather than new samples.
#   -g LIST   GPU indices, default all

set -eu -o pipefail

FIELDS=index,timestamp,power.draw,power.limit,clocks.current.sm,clocks.current.memory,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory,memory.used,pstate,clocks_throttle_reasons.active

summarize() {
    local csv=$1
    [[ -f $csv ]] || { echo "error: no such file: $csv" >&2; exit 1; }
    python3 - "$csv" <<'PY'
import csv, sys
from collections import defaultdict

def norm(k):
    """` power.draw [W]` -> `power.draw`.

    nvidia-smi pads fields with a leading space and keeps the unit in the
    header even under --format=csv,nounits, and it renamed
    clocks_throttle_reasons -> clocks_event_reasons. Normalising the keys is
    cheaper than pinning to one driver's spelling.
    """
    return (k or "").strip().split(" [")[0]

with open(sys.argv[1]) as fh:
    rows = [{norm(k): v for k, v in r.items()} for r in csv.DictReader(fh)]
rows = [r for r in rows if r.get("index", "").strip().isdigit()]
if not rows:
    print("no samples"); sys.exit(0)

def num(v):
    v = (v or "").strip().split()[0] if (v or "").strip() else ""
    try: return float(v)
    except ValueError: return None

by = defaultdict(list)
for r in rows:
    p = num(r.get("power.draw"))
    if p is not None:
        by[r["index"].strip()].append((p, num(r.get("clocks.current.sm")),
                                       num(r.get("temperature.gpu")),
                                       num(r.get("utilization.gpu"))))

# Sample interval from the first two timestamps of GPU 0, for the energy
# integral. Assumes a fixed interval, which is what this script emits.
ts = [r["timestamp"] for r in rows if r["index"].strip() == sorted(by)[0]]
dt = None
if len(ts) > 1:
    from datetime import datetime
    fmt = "%Y/%m/%d %H:%M:%S.%f"
    try:
        dt = (datetime.strptime(ts[1].strip(), fmt)
              - datetime.strptime(ts[0].strip(), fmt)).total_seconds()
    except ValueError:
        pass

print(f"samples: {len(rows)} rows, {len(by)} GPUs"
      + (f", interval ~{dt:.2f}s, span ~{dt*len(ts):.0f}s" if dt else ""))
print()
hdr = f"{'GPU':>3}  {'mean W':>8} {'max W':>8} {'p95 W':>8}  {'mean SM MHz':>11}  {'max degC':>8}  {'mean util%':>10}"
print(hdr); print("-" * len(hdr))
tot_mean = 0.0
for g in sorted(by, key=lambda x: int(x)):
    pw = sorted(v[0] for v in by[g])
    sm = [v[1] for v in by[g] if v[1] is not None]
    tp = [v[2] for v in by[g] if v[2] is not None]
    ut = [v[3] for v in by[g] if v[3] is not None]
    mean = sum(pw)/len(pw); tot_mean += mean
    p95 = pw[min(len(pw)-1, int(0.95*len(pw)))]
    print(f"{g:>3}  {mean:8.1f} {max(pw):8.1f} {p95:8.1f}  "
          f"{(sum(sm)/len(sm) if sm else 0):11.0f}  {(max(tp) if tp else 0):8.0f}  "
          f"{(sum(ut)/len(ut) if ut else 0):10.1f}")
print("-" * len(hdr))
print(f"{'all':>3}  {tot_mean:8.1f} (sum of per-GPU means, W)")
if dt:
    kwh = tot_mean * dt * len(ts) / 3600 / 1000
    print(f"\nenergy over the sampled span: {kwh:.4f} kWh "
          f"({tot_mean*dt*len(ts)/3600:.1f} Wh), GPUs only -- "
          f"no CPU, NIC, or cooling.")
PY
}

if [[ ${1:-} == --summarize ]]; then
    summarize "${2:?usage: $0 --summarize FILE.csv}"
    exit 0
fi

OUT=""; INTERVAL=1; GPUS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -o) OUT=$2; shift 2 ;;
        -i) INTERVAL=$2; shift 2 ;;
        -g) GPUS=$2; shift 2 ;;
        --) shift; break ;;
        -h|--help) sed -n '23,47p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

OUT=${OUT:-power-$(date +%Y%m%d-%H%M%S).csv}
command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi not found" >&2; exit 1; }

# -l takes whole seconds and silently truncates a fraction (-l 0.5 samples
# every 5s, not every 0.5s). Sub-second rates have to go through -lms.
if [[ $INTERVAL == *.* ]] || [[ $INTERVAL == 0 ]]; then
    MS=$(python3 -c "print(int(round(float('$INTERVAL')*1000)))")
    [[ $MS -lt 100 ]] && { echo "error: interval below 0.1s; NVML repeats samples" >&2; exit 1; }
    SMI_ARGS=(--query-gpu="$FIELDS" --format=csv,nounits -lms "$MS")
else
    SMI_ARGS=(--query-gpu="$FIELDS" --format=csv,nounits -l "$INTERVAL")
fi
[[ -n $GPUS ]] && SMI_ARGS+=(-i "$GPUS")

echo "sampling every ${INTERVAL}s -> $OUT"
nvidia-smi "${SMI_ARGS[@]}" >"$OUT" &
SAMPLER=$!
# Kill the sampler however we leave -- Ctrl-C, the command finishing, or an
# error -- otherwise it outlives the shell and keeps appending.
trap 'kill $SAMPLER 2>/dev/null || true' EXIT INT TERM

if [[ $# -gt 0 ]]; then
    echo "running: $*"
    set +e; "$@"; RC=$?; set -e
    sleep "$INTERVAL"          # let the last sample land
    kill $SAMPLER 2>/dev/null || true
    wait $SAMPLER 2>/dev/null || true
    echo; echo "command exited $RC"; echo
    summarize "$OUT"
    exit $RC
fi

echo "Ctrl-C to stop."
wait $SAMPLER
