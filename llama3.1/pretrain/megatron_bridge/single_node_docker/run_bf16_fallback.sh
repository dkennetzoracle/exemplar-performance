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
# ============================================================================
#  NOT A BENCHMARK CONFIGURATION. Do not compare its output to published
#  numbers, and do not quote it as a recipe result.
# ============================================================================
#
# This is the bring-up path: it gets the recipe to train end to end on a GPU
# whose architecture the container ships no kernels for. Getting there costs
# six deviations, every one of them a slowdown (see WHY below). It exists to
# prove the stack runs and to give a baseline to iterate against; the moment a
# container with kernels for the target arch is available, drop all of this and
# use launch_local.sh directly.
#
# Diagnose first, so you know which of these you actually need:
#   ./check_arch_support.sh <image>
#
# Usage:
#   export LLMB_INSTALL=/mnt/nvme/llmb
#   export HF_TOKEN=$(cat /path/to/hf-token)
#   ./run_bf16_fallback.sh
#
# Any variable below can be overridden from the environment, and anything
# launch_local.sh understands is passed straight through.

set -eu -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${LLMB_INSTALL:?set LLMB_INSTALL to the install root, e.g. /mnt/nvme/llmb}"
: "${HF_TOKEN:?set HF_TOKEN to a token with access to the model repo (see QUICKSTART.md)}"

# --- what to run ------------------------------------------------------------
# llama31_8b rather than the recipe default llama3_8b: the llama3 presets read
# meta-llama/Meta-Llama-3-8B, gated separately from Llama 3.1. Use llama3_8b
# here instead if your token has Llama 3 access.
export MODEL_SIZE=${MODEL_SIZE:-8b}
export MODEL_RECIPE_NAME=${MODEL_RECIPE_NAME:-llama31_8b}
# gb200 because llama31_8b ships no vr200 preset, and for 8B the vr200 presets
# are defined as aliases of the gb200 ones anyway. GPU_TYPE only picks a
# preset; it does not have to name the silicon.
export GPU_TYPE=${GPU_TYPE:-gb200}
# nvfp4 selects the *preset* (parallelism and batch sizing). The precision is
# overridden to bf16 below, so the experiment name will say nvfp4 while the run
# is bf16. llama31_8b has no bf16 preset to select instead.
export DTYPE=${DTYPE:-nvfp4}
export CONFIG_VARIANT=${CONFIG_VARIANT:-v1}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-4}
export MAX_STEPS=${MAX_STEPS:-50}
# MBS=1 because unfused attention materializes the full score matrix; see WHY 4.
export MBS=${MBS:-1}
export GBS=${GBS:-8}

# --- WHY each workaround is here -------------------------------------------
# 1. bf16 only. TransformerEngine ships cubins for a fixed arch list and no
#    PTX, so on an unlisted arch every quantized precision dies in its
#    quantize kernel with "no kernel image is available for execution on the
#    device" (fp8/quantize_fp8.cuh, mxfp8/quantize_mxfp8.cuh,
#    hadamard_transform.cu). bf16 lands in cuBLAS/cuDNN, which track the
#    driver, so it survives. bf16_with_nvfp4_mixed differs from bf16_mixed in
#    exactly fp4, fp4_param and fp4_param_gather; this preset already clears
#    the last two, so nulling fp4 is enough.
# 2. No torch.compile. Triton asks ptxas for sm_<cc>a; if that target does not
#    exist, every compiled region fails to build. Megatron's jit_fuser is
#    torch.compile, so TORCHDYNAMO_DISABLE makes those fusions fall back to
#    eager instead of failing.
# 3. Unfused attention. cuDNN has no fused-attention execution plan for the
#    arch ("No valid execution plans built").
# 4. Recompute + MBS=1. Unfused attention materializes
#    mbs x heads x seq x seq x 2 bytes = 8 GiB per tensor at seq 8192, MBS=2,
#    which OOMs a 280 GB GPU. MBS=1 plus full recompute fits.
# 5. No fused cross-entropy. TE implements it in Triton, and it aborts the
#    process with an uncatchable "LLVM ERROR: Cannot select ... shfl.sync" at
#    the first training step.
# 6. COMPAT_SHIM. Unrelated to the arch: Megatron-LM returns grad_norm as a
#    0-dim tensor on some paths, Megatron-Bridge's training_log formats it with
#    a float spec, and training dies at the first logged iteration with
#    TypeError. The shim patches the formatting, which cannot change a computed
#    value. Config-level avoidance does not work.
export COMPAT_SHIM=true

FALLBACK_ENV="TORCHDYNAMO_DISABLE=1 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=1"
export EXTRA_ENV="$FALLBACK_ENV ${EXTRA_ENV:-}"

FALLBACK_OVERRIDES="mixed_precision.fp4=null"
FALLBACK_OVERRIDES+=" mixed_precision.fp8_dot_product_attention=false"
FALLBACK_OVERRIDES+=" model.fp8_dot_product_attention=false"
FALLBACK_OVERRIDES+=" model.use_transformer_engine_op_fuser=false"
FALLBACK_OVERRIDES+=" model.cross_entropy_loss_fusion=false"
FALLBACK_OVERRIDES+=" model.recompute_granularity=full"
FALLBACK_OVERRIDES+=" model.recompute_method=uniform"
FALLBACK_OVERRIDES+=" model.recompute_num_layers=1"
export EXTRA_HYDRA_OVERRIDES="$FALLBACK_OVERRIDES ${EXTRA_HYDRA_OVERRIDES:-}"

cat <<'BANNER'
===================================================================
 bf16 fallback launch -- BRING-UP ONLY, NOT A BENCHMARK
 bf16 (not fp8/nvfp4), eager activations, unfused attention,
 MBS=1, full recompute. Every one of these is a slowdown.
===================================================================
BANNER

exec "$SCRIPT_DIR/launch_local.sh" "$@"
