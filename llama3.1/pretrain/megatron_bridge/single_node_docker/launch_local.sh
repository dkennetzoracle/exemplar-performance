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
# Single-node, no-Slurm launcher for the Megatron-Bridge Llama3.1 pretrain
# recipe. Drop-in alternative to ../launch.sh with the same env-var interface.
#
# ../launch.sh hands the job to Nemo-Run's SlurmExecutor, which builds an sbatch
# script, wraps each task in `srun ... numactl ...`, and applies PerfEnvPlugin to
# the container environment. None of that is reachable without Slurm, so this
# script does the same four things directly:
#
#   1. resolves the perf environment + Hydra overrides (perf_env_local.py)
#   2. starts the NeMo container with `docker run`
#   3. fans out ranks with `torchrun` instead of `srun`
#   4. binds each rank to its NUMA node (rank_wrapper_local.sh)
#
# The rank-local entrypoint is unchanged: the same
# `scripts/performance/run_script.py` from the pinned Megatron-Bridge checkout,
# bind-mounted at its host path exactly as the Slurm path mounts it, with the
# `megatron.bridge` library coming from the container.
#
# Usage:
#   LLMB_INSTALL=/mnt/nvme/llmb JOB_TOTAL_GPUS=4 GPU_TYPE=vr200 ./launch_local.sh

if [ "${BASH_VERSION:0:1}" -lt 4 ] || { [ "${BASH_VERSION:0:1}" -eq 4 ] && [ "${BASH_VERSION:2:1}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(dirname "$SCRIPT_DIR")

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=llama3.1

# Track the framework version the recipe pins rather than hardcoding it here.
FW_VERSION=$(sed -n 's/^export FW_VERSION=\(.*\)$/\1/p' "$RECIPE_DIR/launch.sh")
: "${FW_VERSION:?could not read FW_VERSION from $RECIPE_DIR/launch.sh}"

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL is a required variable (e.g. /mnt/nvme/llmb)}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
MBRIDGE_DIR=${MBRIDGE_DIR:-$LLMB_WORKLOAD/Megatron-Bridge}
IMAGE=${RUN_CONF_IMAGE:-nvcr.io/nvidia/nemo:$FW_VERSION}

export HF_HOME=${HF_HOME:-$LLMB_INSTALL/.cache/huggingface}
export NEMO_HOME=${NEMO_HOME:-$LLMB_INSTALL/.cache/nemo}

DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
CONFIG_VARIANT=${CONFIG_VARIANT:-v2}
CONFIG_VARIANT=${CONFIG_VARIANT,,}
FP8_RECIPE=${FP8_RECIPE:-cs}
FP8_RECIPE=${FP8_RECIPE,,}
# Deviates from ../launch.sh (405b): 405b cannot run on a single node.
MODEL_SIZE=${MODEL_SIZE:-8b}
MODEL_SIZE=${MODEL_SIZE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
PYTORCH_PROFILE_ENABLED=${ENABLE_PYTORCH_PROFILE:-false}
PYTORCH_PROFILE_ENABLED=${PYTORCH_PROFILE_ENABLED,,}
ENABLED_GPU_METRICS=${ENABLE_GPU_METRICS:-false}
ENABLED_GPU_METRICS=${ENABLED_GPU_METRICS,,}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING:-false}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING,,}
PROFILE_START_STEP=${PROFILE_START_STEP:-45}
PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}
DETERMINISTIC=${DETERMINISTIC:-false}
DETERMINISTIC=${DETERMINISTIC,,}
OFFLINE=${OFFLINE:-false}
OFFLINE=${OFFLINE,,}

MAX_STEPS=${MAX_STEPS:-50}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-$MAX_STEPS}

DRYRUN=${DRYRUN:-false}
DRYRUN=${DRYRUN,,}
PRINT_ONLY=${PRINT_ONLY:-false}
PRINT_ONLY=${PRINT_ONLY,,}
MASTER_PORT=${MASTER_PORT:-29500}

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

# --- Preflight ---------------------------------------------------------------
for tool in docker python3 nvidia-smi; do
    command -v "$tool" >/dev/null || {
        echo "error: '$tool' not found on PATH" >&2
        exit 1
    }
done

if [[ ! -f $MBRIDGE_DIR/scripts/performance/run_script.py ]]; then
    echo "error: $MBRIDGE_DIR/scripts/performance/run_script.py not found." >&2
    echo "       Run ./setup_local.sh (or set MBRIDGE_DIR) first." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "error: container image '$IMAGE' not present locally." >&2
    echo "       Run ./setup_local.sh, or: docker pull $IMAGE" >&2
    exit 1
fi

PHYSICAL_GPUS=$(nvidia-smi -L | wc -l)
if ((JOB_TOTAL_GPUS > PHYSICAL_GPUS)); then
    echo "error: JOB_TOTAL_GPUS=$JOB_TOTAL_GPUS but only $PHYSICAL_GPUS GPUs are visible." >&2
    echo "       This launcher is single-node only; use ../launch.sh for multi-node." >&2
    exit 1
fi

# Set model family and recipe names based on model size (mirrors ../launch.sh)
MODEL_FAMILY_NAME="llama"
case $MODEL_SIZE in
    405b) DEFAULT_MODEL_RECIPE_NAME="llama31_405b" ;;
    70b) DEFAULT_MODEL_RECIPE_NAME="llama3_70b" ;;
    8b) DEFAULT_MODEL_RECIPE_NAME="llama3_8b" ;;
    *)
        echo "Error: Unsupported MODEL_SIZE: $MODEL_SIZE" >&2
        exit 1
        ;;
esac
# Escape hatch for the Megatron-Bridge config to load. The 8B/70B defaults are
# the llama3 recipes (see ../README.md), which read the separately gated Llama 3
# repos; `MODEL_RECIPE_NAME=llama31_8b` selects the genuine Llama 3.1 8B recipe
# instead, which reads meta-llama/Meta-Llama-3.1-8B.
MODEL_RECIPE_NAME=${MODEL_RECIPE_NAME:-$DEFAULT_MODEL_RECIPE_NAME}

if [[ $DTYPE == "fp8" ]]; then
    # H100 supports only FP8 CS (mx not allowed)
    if [[ $GPU_TYPE == "h100" ]] && [[ $FP8_RECIPE == "mx" ]]; then
        echo "Error: H100 supports only FP8 CS; FP8 MX is not allowed for this GPU type." >&2
        exit 1
    fi
    case "$FP8_RECIPE" in
        cs | mx) COMPUTE_DTYPE="fp8_$FP8_RECIPE" ;;
        *)
            echo "Error: Other FP8 types are not allowed" >&2
            exit 1
            ;;
    esac
else
    COMPUTE_DTYPE=$DTYPE
fi

if [[ $PROFILE_ENABLED == "true" ]] && [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    echo "Error: ENABLE_PROFILE and ENABLE_PYTORCH_PROFILE are mutually exclusive." >&2
    exit 1
fi

# The recipe is validated at >= 8 GPUs (see ../metadata.yaml). Smaller scales
# still run; they are just not comparable to published numbers.
SUPPORTED_SCALES=$(
    python3 - "$RECIPE_DIR/metadata.yaml" "$GPU_TYPE" "$MODEL_SIZE" <<'PY'
import sys, yaml
meta = yaml.safe_load(open(sys.argv[1]))
gpu_cfg = meta["run"]["gpu_configs"].get(sys.argv[2])
if not gpu_cfg:
    print("")
    sys.exit(0)
for entry in gpu_cfg["model_configs"]:
    if entry["model_size"] == sys.argv[3]:
        print(",".join(str(s) for s in entry["scales"]))
        break
else:
    print("")
PY
)
if [[ -z $SUPPORTED_SCALES ]]; then
    echo "NOTE: ../metadata.yaml has no entry for GPU_TYPE=$GPU_TYPE MODEL_SIZE=$MODEL_SIZE."
    echo "      Proceeding; results are not comparable to published numbers."
elif [[ ",$SUPPORTED_SCALES," != *",$JOB_TOTAL_GPUS,"* ]]; then
    echo "NOTE: JOB_TOTAL_GPUS=$JOB_TOTAL_GPUS is outside the validated scales for"
    echo "      $GPU_TYPE/$MODEL_SIZE ($SUPPORTED_SCALES). Proceeding anyway;"
    echo "      results are not comparable to published numbers."
fi

# --- Resolve the config, env and experiment name -----------------------------
RESOLVE_ARGS=(
    --mbridge-dir "$MBRIDGE_DIR"
    --model_family_name "$MODEL_FAMILY_NAME"
    --model_recipe_name "$MODEL_RECIPE_NAME"
    --gpu "$GPU_TYPE"
    --compute_dtype "$COMPUTE_DTYPE"
    --task pretrain
    --config_variant "$CONFIG_VARIANT"
    --num_gpus "$JOB_TOTAL_GPUS"
    --nemo_home "$NEMO_HOME"
)
[[ -n ${TP:-} ]] && RESOLVE_ARGS+=(--tensor_model_parallel_size "$TP")
[[ -n ${PP:-} ]] && RESOLVE_ARGS+=(--pipeline_model_parallel_size "$PP")
[[ -n ${CP:-} ]] && RESOLVE_ARGS+=(--context_parallel_size "$CP")
[[ -n ${VP:-} ]] && RESOLVE_ARGS+=(--virtual_pipeline_model_parallel_size "$VP")
[[ -n ${EP:-} ]] && RESOLVE_ARGS+=(--expert_model_parallel_size "$EP")
[[ -n ${MBS:-} ]] && RESOLVE_ARGS+=(--micro_batch_size "$MBS")
[[ -n ${GBS:-} ]] && RESOLVE_ARGS+=(--global_batch_size "$GBS")
[[ -n ${HF_TOKEN:-} ]] && RESOLVE_ARGS+=(--hf-token-in-env)
[[ $OFFLINE == true ]] && RESOLVE_ARGS+=(--offline)
[[ $DETERMINISTIC == true ]] && RESOLVE_ARGS+=(--deterministic)

TIMESTAMP=$(date '+%s')
STAGING_ENV_FILE=$(mktemp)
trap 'rm -f "$STAGING_ENV_FILE"' EXIT

# One resolver call settles the experiment name, NUMA divisor, plan, Hydra
# overrides and env file, so any upstream config warning is printed once.
eval "$(python3 "$SCRIPT_DIR/perf_env_local.py" "${RESOLVE_ARGS[@]}" \
    --emit all --env-file "$STAGING_ENV_FILE")"
: "${EXP_NAME:?resolver did not return EXP_NAME}"

# Keep the overrides in an array: values like
# `profiling.profile_ranks=[0,1,2,3]` are glob patterns and must never go
# through unquoted word-splitting.
read -ra HYDRA_OVERRIDES <<<"$HYDRA"

# Same layout the recipe README documents for the Slurm path, so the
# experiments/ tree stays interchangeable.
RUN_DIR=$LLMB_WORKLOAD/experiments/$EXP_NAME/${EXP_NAME}_${TIMESTAMP}/$EXP_NAME
LOG_FILE=$RUN_DIR/log-$EXP_NAME.out

EXTRA_MOUNTS=()

# Checkpoint load runs a single step on top of the restored state, so it has to
# settle MAX_STEPS before the rank-local args are built.
if [[ -n ${LOAD_CHECKPOINT_PATH:-} ]]; then
    if [[ $MODEL_SIZE != 8b ]]; then
        echo "Error: Checkpoint load is only supported for 8B. Current MODEL_SIZE=$MODEL_SIZE." >&2
        exit 1
    fi
    if [[ ! -d $LOAD_CHECKPOINT_PATH ]]; then
        echo "error: LOAD_CHECKPOINT_PATH '$LOAD_CHECKPOINT_PATH' is not a directory" >&2
        exit 1
    fi
    LOAD_CHECKPOINT_PATH=$(cd "$LOAD_CHECKPOINT_PATH" && pwd)
    MAX_STEPS=1
    EXTRA_MOUNTS+=(-v "$LOAD_CHECKPOINT_PATH:$LOAD_CHECKPOINT_PATH")
fi

# --- Build the rank-local command --------------------------------------------
IN_CONTAINER_SCRIPT_DIR=$MBRIDGE_DIR/scripts/performance
RUN_ARGS=(
    --model_family_name "$MODEL_FAMILY_NAME"
    --model_recipe_name "$MODEL_RECIPE_NAME"
    --gpu "$GPU_TYPE"
    --compute_dtype "$COMPUTE_DTYPE"
    --config_variant "$CONFIG_VARIANT"
    --task pretrain
    --data mock
    --num_gpus "$JOB_TOTAL_GPUS"
    --max_steps "$MAX_STEPS"
)
[[ -n ${TP:-} ]] && RUN_ARGS+=(--tensor_model_parallel_size "$TP")
[[ -n ${PP:-} ]] && RUN_ARGS+=(--pipeline_model_parallel_size "$PP")
[[ -n ${CP:-} ]] && RUN_ARGS+=(--context_parallel_size "$CP")
[[ -n ${VP:-} ]] && RUN_ARGS+=(--virtual_pipeline_model_parallel_size "$VP")
[[ -n ${EP:-} ]] && RUN_ARGS+=(--expert_model_parallel_size "$EP")
[[ -n ${MBS:-} ]] && RUN_ARGS+=(--micro_batch_size "$MBS")
[[ -n ${GBS:-} ]] && RUN_ARGS+=(--global_batch_size "$GBS")
# CUDA-graph mode is a preset property, but capture is fragile on unproven
# stacks, so expose the same escape hatch scripts/performance's own interactive
# launcher has. CG_IMPL=none disables graphs entirely.
[[ -n ${CG_IMPL:-} ]] && RUN_ARGS+=(--cuda_graph_impl "$CG_IMPL")
[[ -n ${CG_SCOPE:-} ]] && RUN_ARGS+=(--cuda_graph_scope "$CG_SCOPE")

[[ -n ${LOAD_CHECKPOINT_PATH:-} ]] && RUN_ARGS+=(--load_dir "$LOAD_CHECKPOINT_PATH")

# Checkpointing (8B only, mirroring ../launch.sh)
if [[ $ENABLE_CHECKPOINT == true ]]; then
    if [[ $MODEL_SIZE == 70b ]] || [[ $MODEL_SIZE == 405b ]]; then
        echo "Error: Checkpointing is not supported for 70B or 405B due to a known NCCL error during checkpoint save." >&2
        exit 1
    fi
    if [[ -n ${CHECKPOINT_DIR:-} ]]; then
        mkdir -p "$CHECKPOINT_DIR"
        CHECKPOINT_DIR=$(cd "$CHECKPOINT_DIR" && pwd)
        EXTRA_MOUNTS+=(-v "$CHECKPOINT_DIR:$CHECKPOINT_DIR")
        RUN_ARGS+=(--save_dir "$CHECKPOINT_DIR")
    fi
    RUN_ARGS+=(--save_interval "$CHECKPOINT_INTERVAL")
fi

# Profiling. On Slurm, NsysPlugin both wraps each task in `nsys profile` and
# appends these Hydra overrides. rank_wrapper_local.sh does the wrapping, so
# only the overrides are added here.
DOCKER_CAPS=()
WRAPPER_ENV=()
if [[ $PROFILE_ENABLED == "true" ]]; then
    PROFILE_RANKS=${PROFILE_RANKS:-$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)))}
    HYDRA_OVERRIDES+=(
        "profiling.use_nsys_profiler=true"
        "profiling.profile_step_start=$PROFILE_START_STEP"
        "profiling.profile_step_end=$PROFILE_STOP_STEP"
        "profiling.profile_ranks=[$PROFILE_RANKS]"
        "profiling.record_shapes=false"
    )
    WRAPPER_ENV+=(-e NSYS_ENABLE=1 -e NSYS_OUTPUT_DIR=/nemo_run/nsys_profile)
    [[ $ENABLED_GPU_METRICS == true ]] && WRAPPER_ENV+=(-e NSYS_GPU_METRICS=1)
    # nsys needs these to sample; without them it degrades to a partial trace.
    DOCKER_CAPS+=(--cap-add=SYS_ADMIN --cap-add=SYS_PTRACE)
fi

if [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    PROFILE_RANKS=${PROFILE_RANKS:-$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)))}
    RUN_ARGS+=(--pytorch_profiler true)
    HYDRA_OVERRIDES+=(
        "profiling.use_pytorch_profiler=true"
        "profiling.profile_step_start=$PROFILE_START_STEP"
        "profiling.profile_step_end=$PROFILE_STOP_STEP"
        "profiling.profile_ranks=[$PROFILE_RANKS]"
        "profiling.record_memory_history=false"
        "profiling.memory_snapshot_path=/nemo_run/pytorch_profile/snapshot.pickle"
        "profiling.record_shapes=false"
    )
fi

if [[ $DRYRUN == true ]]; then
    RUN_ARGS+=(--dryrun --save_config_filepath /nemo_run/configs/ConfigContainer.yaml)
fi

# Arbitrary extra Hydra overrides, e.g.
#   EXTRA_HYDRA_OVERRIDES="model.use_transformer_engine_op_fuser=false"
# run_script.py forwards unknown args to set_cli_overrides().
if [[ -n ${EXTRA_HYDRA_OVERRIDES:-} ]]; then
    read -ra _extra <<<"$EXTRA_HYDRA_OVERRIDES"
    HYDRA_OVERRIDES+=("${_extra[@]}")
fi

if [[ -n ${RUN_CONF_MOUNTS:-} ]]; then
    IFS=',' read -ra _mounts <<<"$RUN_CONF_MOUNTS"
    for m in "${_mounts[@]}"; do
        [[ -n $m ]] && EXTRA_MOUNTS+=(-v "$m:$m")
    done
fi

# --- Assemble and run ---------------------------------------------------------
mkdir -p "$RUN_DIR/configs" "$HF_HOME" "$NEMO_HOME"
ENV_FILE=$RUN_DIR/env.list
mv "$STAGING_ENV_FILE" "$ENV_FILE"
trap - EXIT

# Host-specific env additions, e.g.
#   EXTRA_ENV="TORCHDYNAMO_DISABLE=1 NCCL_DEBUG=INFO"
# Appended last so they win over the resolved values, and recorded in env.list
# alongside them.
if [[ -n ${EXTRA_ENV:-} ]]; then
    for kv in $EXTRA_ENV; do
        if [[ $kv != *=* ]]; then
            echo "error: EXTRA_ENV entry '$kv' is not KEY=VALUE" >&2
            exit 1
        fi
        printf '%s\n' "$kv" >>"$ENV_FILE"
    done
fi

DEVICES=$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)))
CONTAINER_NAME=${CONTAINER_NAME:-llmb-$EXP_NAME-$TIMESTAMP}

PROXY_ENV=()
for var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    [[ -n ${!var:-} ]] && PROXY_ENV+=(-e "$var")
done

DOCKER_CMD=(
    docker run --rm
    --name "$CONTAINER_NAME"
    # The inner quotes are literal and required: without them docker splits the
    # comma list and rejects it as "both Count and DeviceIDs".
    --gpus "\"device=$DEVICES\""
    --ipc=host
    --network=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    # numactl's set_mempolicy/cpunodebind need CAP_SYS_NICE, which Docker drops
    # by default. Without it the NUMA binding silently degrades.
    --cap-add=SYS_NICE
    ${DOCKER_CAPS[@]+"${DOCKER_CAPS[@]}"}
    --env-file "$ENV_FILE"
    -e PYTHONPATH="$IN_CONTAINER_SCRIPT_DIR"
    -e HF_HOME="$HF_HOME"
    -e NUMA_DIVISOR="$NUMA_DIVISOR"
    -e ENABLE_PCT_BINDING="$ENABLE_PCT_BINDING"
    ${WRAPPER_ENV[@]+"${WRAPPER_ENV[@]}"}
    # Forwarded by name so a corporate proxy reaches HuggingFace in online mode.
    ${PROXY_ENV[@]+"${PROXY_ENV[@]}"}
    # Bind-mount the pinned scripts/performance at its host path, matching the
    # Slurm executor's `{SCRIPT_DIR}:{SCRIPT_DIR}` mount.
    -v "$IN_CONTAINER_SCRIPT_DIR:$IN_CONTAINER_SCRIPT_DIR:ro"
    -v "$SCRIPT_DIR/rank_wrapper_local.sh:/opt/rank_wrapper_local.sh:ro"
    -v "$SCRIPT_DIR/compat_shim_local.py:/opt/compat_shim_local.py:ro"
    -v "$RUN_DIR:/nemo_run"
    -v "$HF_HOME:$HF_HOME"
    -v "$NEMO_HOME:$NEMO_HOME"
    ${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"}
    -w /opt/Megatron-Bridge
)
# HF_TOKEN is forwarded by name so the value never lands in $ENV_FILE.
[[ -n ${HF_TOKEN:-} ]] && DOCKER_CMD+=(-e HF_TOKEN)

ENTRYPOINT_CMD=(python)
# COMPAT_SHIM=true inserts compat_shim_local.py ahead of the real entrypoint to
# apply library-bug workarounds (see its docstring). Off by default so the
# normal path runs stock upstream code.
if [[ ${COMPAT_SHIM:-false} == true ]]; then
    ENTRYPOINT_CMD+=(/opt/compat_shim_local.py)
fi

TORCHRUN_CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node="$JOB_TOTAL_GPUS"
    --master_addr=127.0.0.1
    --master_port="$MASTER_PORT"
    --no-python
    bash /opt/rank_wrapper_local.sh
    "${ENTRYPOINT_CMD[@]}" "$IN_CONTAINER_SCRIPT_DIR/run_script.py"
    "${RUN_ARGS[@]}"
    ${HYDRA_OVERRIDES[@]+"${HYDRA_OVERRIDES[@]}"}
)

cat <<EOF
===================================================================
 Megatron-Bridge single-node Docker launch (no Slurm)
===================================================================
 Workload:     ${WORKLOAD_TYPE}_${MODEL_NAME} / $MODEL_RECIPE_NAME
 Experiment:   $EXP_NAME
 GPU type:     $GPU_TYPE  (config selection only)
 GPUs:         $JOB_TOTAL_GPUS of $PHYSICAL_GPUS  (nproc_per_node=$JOB_TOTAL_GPUS)
 Precision:    $COMPUTE_DTYPE   Variant: $CONFIG_VARIANT
 Plan:         $PLAN
 Max steps:    $MAX_STEPS
 Image:        $IMAGE
 NUMA:         cpunodebind=LOCAL_RANK/$NUMA_DIVISOR
 Run dir:      $RUN_DIR
 Log:          $LOG_FILE
===================================================================
EOF

if [[ $PRINT_ONLY == true ]]; then
    printf '%q ' "${DOCKER_CMD[@]}" "$IMAGE" "${TORCHRUN_CMD[@]}"
    printf '\n'
    exit 0
fi

set +e
"${DOCKER_CMD[@]}" "$IMAGE" "${TORCHRUN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

echo
echo "Log: $LOG_FILE"
if [[ $STATUS -ne 0 ]]; then
    echo "Training exited with status $STATUS" >&2
    exit "$STATUS"
fi

if [[ $DRYRUN != true ]]; then
    python3 "$SCRIPT_DIR/parse_results_local.py" "$LOG_FILE" || true
fi
