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
# Per-rank wrapper, run inside the container by `torchrun --no-python`.
#
# Replaces the two things srun does per task on the Slurm path:
#   * NUMA/CPU binding -- utils/executors.py builds
#       `numactl --cpunodebind=$((SLURM_LOCALID/N)) --membind=$((SLURM_LOCALID/N))`
#     and prepends it to every task. torchrun exports LOCAL_RANK instead of
#     SLURM_LOCALID, so that is what we divide.
#   * Nsight per-rank capture -- Nemo-Run's launcher wraps each task in `nsys
#     profile` with a rank-stamped output name. Here NSYS_ENABLE=1 does it.
#
# Usage: rank_wrapper_local.sh <command> [args...]

set -eu -o pipefail

: "${LOCAL_RANK:?LOCAL_RANK must be set (this script is meant to run under torchrun --no-python)}"

NUMA_DIVISOR=${NUMA_DIVISOR:-2}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING:-false}
NSYS_ENABLE=${NSYS_ENABLE:-0}
NSYS_OUTPUT_DIR=${NSYS_OUTPUT_DIR:-/nemo_run/nsys_profile}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
NSYS_GPU_METRICS=${NSYS_GPU_METRICS:-0}

numa_node=$((LOCAL_RANK / NUMA_DIVISOR))

BIND_CMD=(numactl --cpunodebind="$numa_node" --membind="$numa_node")
if [[ ${ENABLE_PCT_BINDING,,} == true ]]; then
    # Mirrors the b300 physical-core pinning added in utils/executors.py.
    BIND_CMD+=(-C "$((LOCAL_RANK * 16)),$((LOCAL_RANK * 16 + 1))")
fi

NSYS_CMD=()
if [[ $NSYS_ENABLE == 1 ]]; then
    mkdir -p "$NSYS_OUTPUT_DIR"
    # Slurm names these profile_%p_%q{SLURM_JOB_ID}_node%q{SLURM_NODEID}_rank%q{SLURM_PROCID};
    # single node means node0 and RANK == PROCID.
    NSYS_CMD=(
        nsys profile
        -s none
        -t "$NSYS_TRACE"
        -o "${NSYS_OUTPUT_DIR}/profile_%p_node0_rank${RANK:-$LOCAL_RANK}"
        --force-overwrite true
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        --nvtx-domain-include=NCCL
    )
    if [[ $NSYS_GPU_METRICS == 1 ]]; then
        NSYS_CMD+=(--gpu-metrics-devices=all)
    fi
fi

exec "${BIND_CMD[@]}" "${NSYS_CMD[@]}" "$@"
