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
# Perf matrix driver. Run ON the GPU node, after node_prep.sh + setup_images.sh.
#
#   ./run_matrix.sh reference   # bf16-fallback rows: the only config that runs
#                               # on an arch with no TE cubins. Comparable
#                               # across machines -- this is the VR200 baseline.
#   ./run_matrix.sh native      # native quantized rows (nvfp4 / fp8). Needs TE
#                               # cubins for the arch, so these are the rows an
#                               # unsupported arch cannot run at all.
#   ./run_matrix.sh portable    # arch-INDEPENDENT levers on top of the
#                               # fallback: MBS, recompute, GBS, container.
#                               # Runs anywhere; use it to tune a machine that
#                               # is stuck on the fallback path.
#   ./run_matrix.sh all
#
# Rows are skipped if $RESULTS_DIR/<tag>.done exists, so the script is
# resumable and safe to re-run after fixing one failure.
#
# Env:
#   IMAGE_TAG      nemo tag to run (default: the recipe's FW_VERSION pin)
#   LLMB_INSTALL   install root (default /mnt/localdisk/llmb)
#   RESULTS_DIR    where logs/markers land (default $LLMB_INSTALL/perf-matrix-results)
#   MAX_STEPS      default 50 (the parser averages iters 35-44, so do not lower)
#   JOB_TOTAL_GPUS default 4

set -uo pipefail   # NOT -e: a failing row must not abort the sweep
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

MODE=${1:-}
case $MODE in
    reference | native | portable | all) ;;
    *) echo "usage: $0 {reference|native|portable|all}" >&2; exit 2 ;;
esac

FW_VERSION=$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?FW_VERSION=\(.*\)$/\2/p' \
    "$SND_DIR/../launch.sh" | head -1)
IMAGE_TAG=${IMAGE_TAG:-$FW_VERSION}
IMG=nvcr.io/nvidia/nemo:$IMAGE_TAG
T=${IMAGE_TAG//./}          # tag suffix for row names, e.g. 260601

# Per-image quirk: 26.08.00 crashes in the post-training eval on CUDA-graph
# capture. Skipping eval is harmless elsewhere and keeps exit codes meaningful.
EVAL_OVR=$NO_EVAL

# COMPAT_SHIM is unrelated to the arch: Megatron-LM returns grad_norm as a
# 0-dim tensor on some paths and Megatron-Bridge formats it with a float spec,
# so training dies at the first logged iteration with TypeError. The shim
# patches the formatting only; it cannot change a computed value.
COMMON="RUN_CONF_IMAGE=$IMG COMPAT_SHIM=true"

# llama31_8b, not the MODEL_SIZE=8b default llama3_8b: the latter reads the
# separately-gated Meta-Llama-3-8B. See setup_images.sh's access check.
L8="MODEL_SIZE=8b MODEL_RECIPE_NAME=llama31_8b"
Q30="RECIPE_DIR=$REPO_ROOT/qwen3/pretrain MODEL_SIZE=30b"

echo "==================================================================="
echo " perf matrix: $MODE"
echo " host:    $(hostname)   GPUs: $JOB_TOTAL_GPUS"
echo " image:   $IMG"
echo " results: $RESULTS_DIR"
echo "==================================================================="
echo

# ===========================================================================
# reference -- the bf16-fallback rows.
#
# GPU_TYPE selects a *preset* (parallelism + batch shape); it does not have to
# name the silicon. llama31_8b ships no vr200/gb300 bf16 preset at all
# ("Available variants: []"), so the nvfp4 preset is selected and the precision
# overridden to bf16 -- which is why the experiment name says nvfp4 while the
# run is bf16. Do not compare these to published 8B figures.
# ===========================================================================
run_reference () {
    run ref-$T-llama31-8b-bf16-mbs1 $COMMON $L8 \
        GPU_TYPE=gb200 DTYPE=nvfp4 CONFIG_VARIANT=v1 MBS=1 GBS=8 \
        EXTRA_ENV="$FALLBACK_ENV" \
        EXTRA_HYDRA_OVERRIDES="$FALLBACK_OVERRIDES_BASE $RECOMPUTE_FULL $EVAL_OVR" \
        ./launch_local.sh

    # Qwen3 needs two fewer workarounds than llama (Qwen/Qwen3-30B-A3B is
    # ungated and 30b has a real bf16 preset) but adds the MoE ones. EP=4
    # because the preset is 8 GPUs with EP=8 and EP cannot exceed the ranks
    # available. MOE_BACKEND=alltoall because the preset picks HybridEP, which
    # assumes an NVL72 domain; note the guard is `major in [8,9,10]`, so on a
    # major-10 part HybridEP does engage and must be turned off explicitly.
    # No recompute here: qwen3's preset uses cuda_graph_impl=transformer_engine
    # and full recompute asserts "only supported with full iteration CUDA graph".
    run ref-$T-qwen3-30b-bf16-mbs1 $COMMON $Q30 \
        GPU_TYPE=vr200 DTYPE=bf16 CONFIG_VARIANT=v1 EP=4 MOE_BACKEND=alltoall \
        MBS=1 GBS=256 \
        EXTRA_ENV="$FALLBACK_ENV" \
        EXTRA_HYDRA_OVERRIDES="$FALLBACK_OVERRIDES_BASE $MOE_OVERRIDES $EVAL_OVR" \
        ./launch_local.sh
}

# ===========================================================================
# native -- quantized precision, no arch workarounds beyond NO_FP8_DPA.
#
# MBS ceiling on 4 GPUs is 4 for llama3.1 8B: MBS=8 OOMs at both nvfp4 and
# fp8_cs (activation memory > 276 GiB/GPU). For qwen3 the ceiling is lower
# still because the gb300 preset assumes EP=8 across 8 GPUs, so EP=4 on one
# node doubles expert weights per GPU -- preset MBS=8 OOMs at 267.95/276.62 GiB.
# ===========================================================================
run_native () {
    local NATIVE_OVR="$NO_FP8_DPA $EVAL_OVR"

    # nvfp4: MBS sweep at GA=1, then GA sweep at the memory-feasible MBS.
    for mbs in 2 4; do
        run nat-$T-llama31-8b-nvfp4-mbs$mbs-ga1 $COMMON $L8 \
            GPU_TYPE=gb300 DTYPE=nvfp4 CONFIG_VARIANT=v1 MBS=$mbs GBS=$((mbs*4)) \
            EXTRA_HYDRA_OVERRIDES="$NATIVE_OVR" ./launch_local.sh
    done
    for gbs in 64 128 256; do
        run nat-$T-llama31-8b-nvfp4-mbs4-gbs$gbs $COMMON $L8 \
            GPU_TYPE=gb300 DTYPE=nvfp4 CONFIG_VARIANT=v1 MBS=4 GBS=$gbs \
            EXTRA_HYDRA_OVERRIDES="$NATIVE_OVR" ./launch_local.sh
    done

    # fp8_cs. The gb300 fp8 preset is an 8-GPU shape (TP*PP*CP=8), which cannot
    # shard over 4 GPUs, so parallelism is pinned to 1/1/1 -- a deviation from
    # the preset that must be reported alongside the number.
    for mbs in 2 4; do
        run nat-$T-llama31-8b-fp8cs-mbs$mbs-ga1 $COMMON $L8 \
            GPU_TYPE=gb300 DTYPE=fp8 FP8_RECIPE=cs TP=1 PP=1 CP=1 \
            MBS=$mbs GBS=$((mbs*4)) EXTRA_HYDRA_OVERRIDES="$EVAL_OVR" ./launch_local.sh
    done
    run nat-$T-llama31-8b-fp8cs-mbs4-gbs256 $COMMON $L8 \
        GPU_TYPE=gb300 DTYPE=fp8 FP8_RECIPE=cs TP=1 PP=1 CP=1 MBS=4 GBS=256 \
        EXTRA_HYDRA_OVERRIDES="$EVAL_OVR" ./launch_local.sh

    # config variant v2, for completeness of the preset space.
    run nat-$T-llama31-8b-nvfp4-v2-mbs4 $COMMON $L8 \
        GPU_TYPE=gb300 DTYPE=nvfp4 CONFIG_VARIANT=v2 MBS=4 GBS=16 \
        EXTRA_HYDRA_OVERRIDES="$NATIVE_OVR" ./launch_local.sh

    # Qwen3 30B-A3B MoE.
    for mbs in 1 2 4; do
        run nat-$T-qwen3-30b-bf16-mbs$mbs $COMMON $Q30 \
            GPU_TYPE=gb300 DTYPE=bf16 CONFIG_VARIANT=v1 EP=4 MOE_BACKEND=alltoall \
            MBS=$mbs GBS=256 EXTRA_HYDRA_OVERRIDES="$EVAL_OVR" ./launch_local.sh
        run nat-$T-qwen3-30b-fp8cs-mbs$mbs $COMMON $Q30 \
            GPU_TYPE=gb300 DTYPE=fp8 FP8_RECIPE=cs CONFIG_VARIANT=v1 EP=4 \
            MOE_BACKEND=alltoall MBS=$mbs GBS=256 \
            EXTRA_HYDRA_OVERRIDES="$NATIVE_OVR" ./launch_local.sh
    done
}

# ===========================================================================
# portable -- levers that need NO kernels the container might lack, so they
# apply to an arch stuck on the bf16 fallback (e.g. sm_107 on nemo 26.06.01 /
# 26.08.00, where bf16 is OK but fp8/nvfp4/torch.compile all fail).
#
# All six fallback workarounds stay ON. Only these vary:
#   MBS       1 vs 2  -- QUICKSTART says MBS=1 is required because unfused
#                        attention OOMs at MBS=2. That did NOT reproduce on a
#                        284 GB GB300 with the identical config, so it is worth
#                        re-testing per machine rather than assumed.
#   recompute full vs off -- full recompute was added *because* of unfused
#                        attention's score matrix. It adds ~30% extra compute,
#                        so if it is not actually needed this is the single
#                        biggest portable win.
#   GBS       8 / 32 / 64 -- pure batch-shape economics: a larger GBS amortises
#                        the optimizer step and the DP all-reduce over more
#                        microbatches. No arch dependency.
#   container -- run with IMAGE_TAG=26.08.00 as well; bf16 is still OK there
#                on sm_107 per arch_support/nemo-26.08.00.md.
# ===========================================================================
run_portable () {
    # Qwen3 MBS first: 1 -> 2 measured 1.91x on GB300 (7,860 -> 15,042
    # tok/s/GPU), the largest portable win found, and pure batch shape. GBS is
    # already 256 (GA=64) so MBS is the only lever with headroom here. No
    # recompute row -- see the note in run_reference.
    for mbs in 2 4; do
        run por-$T-qwen3-30b-bf16-mbs$mbs $COMMON $Q30 \
            GPU_TYPE=vr200 DTYPE=bf16 CONFIG_VARIANT=v1 EP=4 MOE_BACKEND=alltoall \
            MBS=$mbs GBS=256 \
            EXTRA_ENV="$FALLBACK_ENV" \
            EXTRA_HYDRA_OVERRIDES="$FALLBACK_OVERRIDES_BASE $MOE_OVERRIDES $EVAL_OVR" \
            ./launch_local.sh
    done

    # llama MBS x recompute. MBS=4 is untested in the fallback path on either
    # machine; MBS dominated every other lever, so it is worth the two runs
    # even if it OOMs.
    for mbs in 1 2 4; do
        # GBS must be a multiple of DP*MBS. The reference GBS=8 covers MBS=1
        # (GA=2) and MBS=2 (GA=1), but MBS=4 needs at least 16. Comparing
        # across the resulting shapes is exactly what tokens/s/GPU is for.
        gbs=$((mbs * JOB_TOTAL_GPUS)); ((gbs < 8)) && gbs=8
        run por-$T-llama31-8b-bf16-mbs$mbs-recompute $COMMON $L8 \
            GPU_TYPE=gb200 DTYPE=nvfp4 CONFIG_VARIANT=v1 MBS=$mbs GBS=$gbs \
            EXTRA_ENV="$FALLBACK_ENV" \
            EXTRA_HYDRA_OVERRIDES="$FALLBACK_OVERRIDES_BASE $RECOMPUTE_FULL $EVAL_OVR" \
            ./launch_local.sh
        run por-$T-llama31-8b-bf16-mbs$mbs-norecompute $COMMON $L8 \
            GPU_TYPE=gb200 DTYPE=nvfp4 CONFIG_VARIANT=v1 MBS=$mbs GBS=$gbs \
            EXTRA_ENV="$FALLBACK_ENV" \
            EXTRA_HYDRA_OVERRIDES="$FALLBACK_OVERRIDES_BASE $RECOMPUTE_OFF $EVAL_OVR" \
            ./launch_local.sh
    done

    # GBS / gradient-accumulation sweep at the reference MBS=1.
    for gbs in 32 64; do
        run por-$T-llama31-8b-bf16-mbs1-gbs$gbs $COMMON $L8 \
            GPU_TYPE=gb200 DTYPE=nvfp4 CONFIG_VARIANT=v1 MBS=1 GBS=$gbs \
            EXTRA_ENV="$FALLBACK_ENV" \
            EXTRA_HYDRA_OVERRIDES="$FALLBACK_OVERRIDES_BASE $RECOMPUTE_FULL $EVAL_OVR" \
            ./launch_local.sh
    done

}

case $MODE in
    reference) run_reference ;;
    native)    run_native ;;
    portable)  run_portable ;;
    all)       run_reference; run_native; run_portable ;;
esac

echo "==================================================================="
echo " $MODE done $(date -Is)"
echo " Summarise with: ./collect_results.py"
echo "==================================================================="
