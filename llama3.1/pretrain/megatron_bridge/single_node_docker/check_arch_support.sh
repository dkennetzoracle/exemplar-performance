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
# Answer "can this container actually train on this GPU?" in about a minute,
# before committing to a 50-step run.
#
# On brand-new silicon a container can be missing kernels in several places at
# once, and each one only surfaces after the previous is worked around. This
# probes all of them up front: the PTX assembler's target list, whether
# TransformerEngine ships cubins for the arch (and whether it has any PTX to JIT
# from), whether each compute precision actually executes, and whether Triton
# can compile at all.
#
# Usage:
#   ./check_arch_support.sh [image] [gpu-index]
#
# Defaults to the image named by FW_VERSION in ../launch.sh, GPU 0.

set -eu -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(dirname "$SCRIPT_DIR")

FW_VERSION=$(sed -n 's/^export FW_VERSION=\(.*\)$/\1/p' "$RECIPE_DIR/launch.sh" 2>/dev/null || true)
IMAGE=${1:-nvcr.io/nvidia/nemo:${FW_VERSION:-dev}}
GPU_INDEX=${2:-0}

command -v docker >/dev/null || {
    echo "error: docker not found" >&2
    exit 1
}
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "error: image '$IMAGE' not present locally; docker pull it first" >&2
    exit 1
}

echo "==================================================================="
echo " Arch support probe"
echo "==================================================================="
echo " Image: $IMAGE"
if command -v nvidia-smi >/dev/null; then
    echo " Host GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i "$GPU_INDEX")"
fi
echo "==================================================================="

docker run --rm --gpus "\"device=$GPU_INDEX\"" \
    -e PYTHONWARNINGS=ignore \
    "$IMAGE" bash -c '
set -u
PTXAS=${PTXAS:-/usr/local/cuda/bin/ptxas}

echo "PROBE|cuda_version|${CUDA_VERSION:-unknown}"
echo "PROBE|torch_version|$(python -c "import torch;print(torch.__version__)" 2>/dev/null)"
echo "PROBE|te_version|$(python -c "import transformer_engine as te;print(te.__version__)" 2>/dev/null)"

# --- capability as the toolchain sees it ---
# No single quotes here: this whole block is inside a single-quoted bash -c.
CC=$(python -c "import torch;a,b=torch.cuda.get_device_capability(0);print(str(a)+str(b))" 2>/dev/null)
echo "PROBE|compute_cap|sm_${CC}"

# --- what ptxas can target. Triton asks for sm_<cc>a specifically. ---
ARCHS=$($PTXAS --help 2>&1 | grep -A16 "Allowed values for this option" \
    | grep -oE "sm_[0-9]+[af]?" | sort -u | tr "\n" " ")
echo "PROBE|ptxas_archs|$ARCHS"
case " $ARCHS " in
    *" sm_${CC}a "*) echo "PROBE|ptxas_target|OK sm_${CC}a is supported" ;;
    *) echo "PROBE|ptxas_target|MISSING sm_${CC}a not in ptxas target list (Triton JIT cannot build)" ;;
esac

# --- TransformerEngine kernel coverage. No PTX means no JIT fallback. ---
SO=$(python -c "
import glob, os, transformer_engine
d=os.path.dirname(transformer_engine.__file__)
c=glob.glob(d+\"/**/libtransformer_engine*.so\", recursive=True)
print(c[0] if c else \"\")" 2>/dev/null)
if [ -n "$SO" ]; then
    TE_ELF=$(/usr/local/cuda/bin/cuobjdump --list-elf "$SO" 2>/dev/null \
        | grep -oE "sm_[0-9]+[af]?" | sort -u | tr "\n" " ")
    TE_PTX=$(/usr/local/cuda/bin/cuobjdump --list-ptx "$SO" 2>/dev/null \
        | grep -coE "compute_[0-9]+" || true)
    echo "PROBE|te_cubin_archs|$TE_ELF"
    echo "PROBE|te_ptx_entries|$TE_PTX (0 means no JIT fallback)"
else
    echo "PROBE|te_cubin_archs|could not locate libtransformer_engine.so"
fi

# --- does each precision actually execute? ---
python - <<PY 2>/dev/null
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as R

a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
lin = te.Linear(512, 512).cuda().to(torch.bfloat16)

def report(name, exc=None, note=""):
    if exc is None:
        print(f"PROBE|precision_{name}|OK{note}")
    else:
        msg = str(exc).strip().splitlines()[-1][:100]
        print(f"PROBE|precision_{name}|FAIL {msg}")

try:
    lin(a); torch.cuda.synchronize(); report("bf16")
except Exception as e:
    report("bf16", e)

for name, attr in [("fp8_cs", "DelayedScaling"), ("fp8_mx", "MXFP8BlockScaling"),
                   ("nvfp4", "NVFP4BlockScaling")]:
    rec_cls = getattr(R, attr, None)
    if rec_cls is None:
        print(f"PROBE|precision_{name}|N/A recipe {attr} not in this TE")
        continue
    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=rec_cls()):
            lin(a)
        torch.cuda.synchronize()
        report(name)
    except Exception as e:
        report(name, e)

# --- can Triton compile anything at all? Megatron jit_fuser is torch.compile. ---
try:
    f = torch.compile(lambda x: torch.nn.functional.silu(x) * x)
    f(a); torch.cuda.synchronize()
    print("PROBE|torch_compile|OK")
except Exception as e:
    print(f"PROBE|torch_compile|FAIL {str(e).strip().splitlines()[-1][:100]}")
PY
' 2>/dev/null | grep '^PROBE|' | awk -F'|' '{printf "  %-18s %s\n", $2, $3}'

echo "==================================================================="
echo " Any MISSING/FAIL above means that layer has no code for this GPU."
echo " Quantized precisions need TE cubins for the arch; torch.compile and"
echo " Megatron's jit_fuser need ptxas to accept sm_<cc>a."
echo "==================================================================="
