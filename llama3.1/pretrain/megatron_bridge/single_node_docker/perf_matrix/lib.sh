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
# Shared helpers for the perf matrix. Sourced by run_matrix.sh; not executable
# on its own.

# Resolve paths relative to this file so the scripts work from any cwd.
PM_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SND_DIR=$(dirname "$PM_DIR")                       # single_node_docker/
REPO_ROOT=$(cd -- "$SND_DIR/../../../.." && pwd)   # repo checkout root

: "${LLMB_INSTALL:=/mnt/localdisk/llmb}"
: "${RESULTS_DIR:=$LLMB_INSTALL/perf-matrix-results}"
: "${MAX_STEPS:=50}"                # parser window is iters 35-44, so 50 is the floor
: "${JOB_TOTAL_GPUS:=4}"
export LLMB_INSTALL MAX_STEPS JOB_TOTAL_GPUS

# The repo's own parser (llmb_run.pretrain_log_parser) needs typer/pydantic/
# rich/zstandard. setup_images.sh builds a venv for it; prefer it if present so
# launch_local.sh's own post-run parse works instead of erroring out.
PARSER_VENV=${PARSER_VENV:-$LLMB_INSTALL/parser-venv}
[[ -x $PARSER_VENV/bin/python ]] && export PATH="$PARSER_VENV/bin:$PATH"

# Credentials. Kept in files so keys never land in shell history or env.list.
: "${HF_TOKEN_FILE:=$REPO_ROOT/.hf}"
: "${NGC_TOKEN_FILE:=$REPO_ROOT/.nv}"
if [[ -z ${HF_TOKEN:-} && -r $HF_TOKEN_FILE ]]; then
    HF_TOKEN=$(tr -d '[:space:]' < "$HF_TOKEN_FILE")
    export HF_TOKEN
fi

# Created lazily by ensure_results_dir() rather than at source time, so that
# sourcing lib.sh on a host without the local NVMe (e.g. a Slurm controller)
# stays quiet -- collect_results.py is often run from there.
ensure_results_dir () {
    mkdir -p "$RESULTS_DIR" 2>/dev/null && return 0
    echo "error: cannot create RESULTS_DIR=$RESULTS_DIR" >&2
    echo "       Set RESULTS_DIR to a writable path, or run this on the GPU node." >&2
    return 1
}

# ---------------------------------------------------------------------------
# The six bf16-fallback workarounds, spelled out rather than delegated to
# run_bf16_fallback.sh, so individual ones can be varied per row. Keep in sync
# with run_bf16_fallback.sh -- see its WHY block for the rationale of each.
# ---------------------------------------------------------------------------
FALLBACK_ENV="TORCHDYNAMO_DISABLE=1 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=1"

# Everything except recompute. Recompute is appended per-row because whether it
# is actually required is one of the things under test.
FALLBACK_OVERRIDES_BASE="mixed_precision.fp4=null"
FALLBACK_OVERRIDES_BASE+=" mixed_precision.fp8_dot_product_attention=false"
FALLBACK_OVERRIDES_BASE+=" model.fp8_dot_product_attention=false"
FALLBACK_OVERRIDES_BASE+=" model.use_transformer_engine_op_fuser=false"
FALLBACK_OVERRIDES_BASE+=" model.cross_entropy_loss_fusion=false"

RECOMPUTE_FULL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
RECOMPUTE_OFF="model.recompute_granularity=null"

# MoE-specific workarounds (Triton call sites beyond the dense model's).
MOE_OVERRIDES="model.moe_permute_fusion=false model.moe_router_fusion=false"

# Every nvfp4/fp8 preset sets fp8_dot_product_attention=true. TransformerEngine
# has no fp8 attention backend for sm_103 (reproduced at MBS=1 and MBS=2 on
# nemo:26.06.01), so the native quantized path needs this one override or it
# dies with "No dot product attention backend is available". Re-test per image:
# if a newer container fixes it, dropping this is itself a result.
NO_FP8_DPA="mixed_precision.fp8_dot_product_attention=false model.fp8_dot_product_attention=false"

# nemo:26.08.00 (torch 2.13.0a0) hits a PyTorch INTERNAL ASSERT in CUDA-graph
# capture during the POST-TRAINING eval; training itself is unaffected. Skipping
# eval keeps exit codes meaningful. Harmless on 26.06.01.
NO_EVAL="train.eval_iters=0"

# ---------------------------------------------------------------------------
# run <tag> <KEY=VALUE ...> -- <entrypoint>
#   Runs one config, logs to $RESULTS_DIR/<tag>.log, prints the parsed metrics,
#   and drops a .done marker so re-running the script skips finished rows.
# ---------------------------------------------------------------------------
run () {
    local tag=$1; shift
    ensure_results_dir || return 1
    local log=$RESULTS_DIR/$tag.log

    if [[ -f $RESULTS_DIR/$tag.done ]]; then
        echo "== SKIP $tag (already done)"
        return 0
    fi

    echo "== $tag  started $(date -Is)"
    ( cd "$SND_DIR" && env MAX_STEPS="$MAX_STEPS" JOB_TOTAL_GPUS="$JOB_TOTAL_GPUS" "$@" ) \
        >"$log" 2>&1
    local st=$?

    echo "== $tag  exit=$st"
    grep -E 's/iter:|TFLOPS/GPU:' "$log" | sed 's/^/   /'

    if [[ $st -eq 0 ]] && grep -q 'TFLOPS/GPU:' "$log"; then
        touch "$RESULTS_DIR/$tag.done"
    else
        # Surface the cause inline; collect_results.py classifies it properly.
        grep -m2 -oE 'OutOfMemoryError: CUDA out of memory|No dot product attention backend is available|no kernel image is available|INTERNAL ASSERT FAILED|is not divisible by TP\*PP\*CP|Failed to get config from module_name' \
            "$log" | sed 's/^/   ! /'
    fi
    echo
}
