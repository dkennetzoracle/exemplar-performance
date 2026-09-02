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
# Produce a per-image arch-support report with full, untruncated errors --
# the artifact to attach to a vendor ticket when a container has no kernels
# for a GPU.
#
# check_arch_support.sh answers "is this image usable?" interactively and
# truncates messages to stay readable. This one is the opposite: it walks
# several images and records everything verbatim, in Markdown.
#
# Each compute precision is probed in its **own container process**. That
# matters: a failed CUDA kernel launch can leave the context in an error
# state, so probing several precisions in one process makes every result after
# the first failure untrustworthy.
#
# One self-contained Markdown file is written per image, so a single file can
# be attached to a ticket without carrying unrelated images along.
#
# Usage:
#   ./collect_arch_evidence.sh [-d outdir] [-g gpu-index] IMAGE [IMAGE...]
#
# Defaults: -d arch_support/ next to this script, -g 0.

set -eu -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTDIR=$SCRIPT_DIR/arch_support
GPU_INDEX=0

while getopts ":d:g:" opt; do
    case $opt in
        d) OUTDIR=$OPTARG ;;
        g) GPU_INDEX=$OPTARG ;;
        *)
            echo "usage: $0 [-d outdir] [-g gpu-index] IMAGE [IMAGE...]" >&2
            exit 2
            ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then
    echo "usage: $0 [-d outdir] [-g gpu-index] IMAGE [IMAGE...]" >&2
    exit 2
fi

mkdir -p "$OUTDIR"

# nvcr.io/nvidia/nemo:26.08.00 -> nemo-26.08.00
image_slug() { basename "$1" | tr ':' '-' | tr -cd '[:alnum:]._-'; }

command -v docker >/dev/null || {
    echo "error: docker not found" >&2
    exit 1
}

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

# Probe one precision and print the full traceback on failure.
cat >"$WORKDIR/probe_precision.py" <<'PY'
import sys
import traceback

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as R

which = sys.argv[1]
cap = torch.cuda.get_device_capability(0)
print(f"@@META te={transformer_engine.__version__} torch={torch.__version__} cap=sm_{cap[0]}{cap[1]}")

a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
lin = te.Linear(512, 512).cuda().to(torch.bfloat16)

RECIPES = {
    "bf16": None,
    "fp8_cs": getattr(R, "DelayedScaling", None),
    "fp8_mx": getattr(R, "MXFP8BlockScaling", None),
    "fp8_ds": getattr(R, "DelayedScaling", None),
    "nvfp4": getattr(R, "NVFP4BlockScaling", None),
}
if which not in RECIPES:
    print(f"@@RESULT {which} SKIP unknown precision")
    sys.exit(0)
cls = RECIPES[which]
if which != "bf16" and cls is None:
    print(f"@@RESULT {which} NA recipe class not present in this TransformerEngine")
    sys.exit(0)

try:
    if cls is None:
        out = lin(a)
    else:
        with te.fp8_autocast(enabled=True, fp8_recipe=cls()):
            out = lin(a)
    torch.cuda.synchronize()
    print(f"@@RESULT {which} OK checksum={float(out.detach().float().sum()):.3f}")
except Exception:
    print(f"@@RESULT {which} FAIL")
    print("@@TRACEBACK")
    traceback.print_exc(file=sys.stdout)
    print("@@ENDTRACEBACK")
PY

# TransformerEngine binary coverage. Kept in a file rather than inlined into
# `docker run bash -c` so the nested quoting cannot mangle the Python.
cat >"$WORKDIR/probe_te_binary.sh" <<'SH'
#!/bin/bash
# Locate the library by filesystem search rather than by importing
# transformer_engine: the import pulls in libcuda.so.1, which is only present
# when the container is started with --gpus, and this probe deliberately does
# not need a GPU.
SO=$(find /opt /usr/local/lib /usr/lib -maxdepth 7 -name 'libtransformer_engine*.so' \
    -not -name '*.so.*' -print 2>/dev/null | head -1)
echo "SO=$SO"
if [ -n "$SO" ] && [ -f "$SO" ]; then
    CUOBJ=/usr/local/cuda/bin/cuobjdump
    echo "ELF=$($CUOBJ --list-elf "$SO" 2>/dev/null | grep -oE 'sm_[0-9]+[af]?' | sort -u | tr '\n' ' ')"
    echo "PTX=$($CUOBJ --list-ptx "$SO" 2>/dev/null | grep -c 'compute_')"
fi
SH

# torch.compile / Triton health, separate process for the same reason.
cat >"$WORKDIR/probe_compile.py" <<'PY'
import sys
import traceback

import torch

a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
try:
    fn = torch.compile(lambda x: torch.nn.functional.silu(x) * x)
    fn(a)
    torch.cuda.synchronize()
    print("@@RESULT torch_compile OK")
except Exception:
    print("@@RESULT torch_compile FAIL")
    print("@@TRACEBACK")
    traceback.print_exc(file=sys.stdout)
    print("@@ENDTRACEBACK")
PY

run_in() { # image, script, args...
    local image=$1 script=$2
    shift 2
    docker run --rm --gpus "\"device=$GPU_INDEX\"" \
        -e PYTHONWARNINGS=ignore \
        -v "$WORKDIR:/probe:ro" \
        "$image" python "/probe/$script" "$@" 2>&1 || true
}

host_gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i "$GPU_INDEX" 2>/dev/null || echo "unknown")
host_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_INDEX" 2>/dev/null || echo "unknown")
host_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU_INDEX" 2>/dev/null | tr -d ' .')

write_header() { # $1 = output file, $2 = image
    {
        echo "# Arch-support report: \`$2\`"
        echo
        echo "Generated by \`collect_arch_evidence.sh\`. Every message is verbatim and"
        echo "untruncated. Each precision was probed in its own container process, because"
        echo "a failed kernel launch can poison the CUDA context and make later probes in"
        echo "the same process unreliable."
        echo
        echo "| host property | value |"
        echo "| --- | --- |"
        echo "| GPU | \`$host_gpu\` |"
        echo "| driver | \`$host_driver\` |"
        echo "| Triton target it implies | \`sm_${host_cc}a\` |"
        echo "| collected (UTC) | \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\` |"
        echo
        echo "Triton derives its target from \`torch.cuda.get_device_capability()\` as"
        echo "\`sm_<major><minor>a\`, so that is the name \`ptxas\` must accept."
        echo
    } >"$1"
}

declare -a WRITTEN=()
for image in "$@"; do
    echo "collecting: $image" >&2
    OUT=$OUTDIR/$(image_slug "$image").md
    write_header "$OUT" "$image"
    WRITTEN+=("$OUT")
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        {
            echo "> Not present locally; skipped. \`docker pull $image\` first."
            echo
        } >>"$OUT"
        continue
    fi

    digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null || echo "n/a")
    meta=$(run_in "$image" probe_precision.py bf16 | grep '^@@META' | head -1 || true)
    te_ver=$(sed -n 's/.*te=\([^ ]*\).*/\1/p' <<<"$meta")
    torch_ver=$(sed -n 's/.*torch=\([^ ]*\).*/\1/p' <<<"$meta")
    cap=$(sed -n 's/.*cap=\([^ ]*\).*/\1/p' <<<"$meta")
    cuda_ver=$(docker run --rm "$image" bash -c 'echo ${CUDA_VERSION:-unknown}' 2>/dev/null | tail -1)

    # ptxas target list, and whether it covers this GPU.
    ptxas_archs=$(docker run --rm "$image" /usr/local/cuda/bin/ptxas --help 2>/dev/null |
        grep -A16 "Allowed values for this option" |
        grep -oE "sm_[0-9]+[af]?" | sort -u | tr '\n' ' ' || true)
    if [[ -n $cap ]] && [[ " $ptxas_archs " == *" ${cap}a "* ]]; then
        ptxas_verdict="**yes**"
    else
        ptxas_verdict="**NO**"
    fi

    # TransformerEngine binary coverage. No PTX means no JIT fallback.
    te_info=$(docker run --rm -v "$WORKDIR:/probe:ro" "$image" \
        bash /probe/probe_te_binary.sh 2>/dev/null | grep -E '^(SO|ELF|PTX)=' || true)
    te_so=$(sed -n 's/^SO=//p' <<<"$te_info")
    te_elf=$(sed -n 's/^ELF=//p' <<<"$te_info")
    te_ptx=$(sed -n 's/^PTX=//p' <<<"$te_info")

    {
        echo "## Image"
        echo
        echo "| property | value |"
        echo "| --- | --- |"
        echo "| digest | \`$digest\` |"
        echo "| \`CUDA_VERSION\` | \`${cuda_ver:-unknown}\` |"
        echo "| TransformerEngine | \`${te_ver:-unknown}\` |"
        echo "| torch | \`${torch_ver:-unknown}\` |"
        echo "| GPU capability seen in container | \`${cap:-unknown}\` |"
        echo
        echo "### Toolchain coverage"
        echo
        echo "| check | value |"
        echo "| --- | --- |"
        echo "| \`ptxas\` accepts \`${cap:-?}a\` | $ptxas_verdict |"
        echo "| \`ptxas\` target list | \`${ptxas_archs:-unavailable}\` |"
        echo "| TE cubin architectures | \`${te_elf:-unavailable}\` |"
        echo "| TE embedded PTX entries | \`${te_ptx:-unavailable}\` (0 = no JIT fallback) |"
        echo
        if [[ ${te_ptx:-1} == 0 ]]; then
            echo "With zero PTX entries the shared object can only run on the exact"
            echo "architectures listed above; there is nothing for the driver to JIT."
            echo
        fi
    } >>"$OUT"

    # --- per-precision, one process each ---
    declare -a rows=()
    declare -a blocks=()
    for prec in bf16 fp8_cs fp8_mx nvfp4; do
        raw=$(run_in "$image" probe_precision.py "$prec")
        line=$(grep '^@@RESULT' <<<"$raw" | head -1 || true)
        verdict=$(awk '{print $3}' <<<"$line")
        detail=$(cut -d' ' -f4- <<<"$line")
        tb=$(awk '/^@@TRACEBACK$/{f=1;next} /^@@ENDTRACEBACK$/{f=0} f' <<<"$raw" || true)
        if [[ -n $tb ]]; then
            # Final exception line is the useful one-liner for a summary table.
            # The informative part (file:line, function, CUDA message) is at the
            # END of these lines, behind a long build path. Drop the build path
            # so the summary column shows the part that matters.
            last=$(grep -E "^[A-Za-z_.]*(Error|Exception)" <<<"$tb" | tail -1 |
                sed 's#/opt/uv_cache/[^ ]*/transformer_engine/#transformer_engine/#' |
                cut -c1-160)
            rows+=("| \`$prec\` | ${verdict:-?} | ${last:-see traceback} |")
            blocks+=("$prec"$'\n'"$tb")
        else
            rows+=("| \`$prec\` | ${verdict:-?} | ${detail:-} |")
        fi
    done

    raw=$(run_in "$image" probe_compile.py)
    line=$(grep '^@@RESULT' <<<"$raw" | head -1 || true)
    verdict=$(awk '{print $3}' <<<"$line")
    tb=$(awk '/^@@TRACEBACK$/{f=1;next} /^@@ENDTRACEBACK$/{f=0} f' <<<"$raw" || true)
    if [[ -n $tb ]]; then
        last=$(grep -E "^[A-Za-z_.]*(Error|Exception)" <<<"$tb" | tail -1 | cut -c1-120)
        rows+=("| \`torch.compile\` | ${verdict:-?} | ${last:-see traceback} |")
        blocks+=("torch.compile"$'\n'"$tb")
    else
        rows+=("| \`torch.compile\` | ${verdict:-?} | |")
    fi

    {
        echo "### Results"
        echo
        echo "| probe | result | error |"
        echo "| --- | --- | --- |"
        printf '%s\n' "${rows[@]}"
        echo
        if [[ ${#blocks[@]} -gt 0 ]]; then
            echo "### Full tracebacks"
            echo
            for b in "${blocks[@]}"; do
                name=$(head -1 <<<"$b")
                body=$(tail -n +2 <<<"$b")
                echo "#### \`$name\`"
                echo
                echo '```'
                printf '%s\n' "$body"
                echo '```'
                echo
            done
        fi
    } >>"$OUT"
    unset rows blocks
done

printf 'wrote %s\n' "${WRITTEN[@]}" >&2
