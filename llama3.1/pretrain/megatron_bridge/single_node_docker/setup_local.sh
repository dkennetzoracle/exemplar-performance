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
# Single-node Docker replacement for `llmb-install` for this one recipe.
#
# Does the three things launch_local.sh needs and nothing else:
#   1. clone Megatron-Bridge at the commit pinned in ../metadata.yaml
#   2. pull the NeMo container named by FW_VERSION in ../launch.sh
#   3. create the HF / NeMo cache dirs (and optionally prefetch the tokenizer)
#
# Usage:
#   LLMB_INSTALL=/mnt/nvme/llmb ./setup_local.sh
#
# Env:
#   LLMB_INSTALL    install root (required)
#   IMAGE           override the container image reference
#   SKIP_PULL       true to skip `docker pull`
#   SKIP_CLONE      true to skip the Megatron-Bridge clone
#   RECIPE_DIR      recipe to install (default: the one this script sits in)
#   MODEL_SIZE      selects which declared HF repo to prefetch (default 8b)
#   PREFETCH_HF     true to pre-download the model config into HF_HOME
#                   (needs HF_TOKEN, and access if the repo is gated)

set -eu -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Defaults to the recipe this script sits inside; point it elsewhere to install
# a different megatron_bridge recipe with the same tooling.
RECIPE_DIR=${RECIPE_DIR:-$(dirname "$SCRIPT_DIR")}
RECIPE_DIR=$(cd -- "$RECIPE_DIR" && pwd)

# Workload identity from the recipe's own metadata, so the install layout
# matches what its Slurm path expects.
read -r WORKLOAD_TYPE MODEL_NAME < <(
    python3 -c '
import sys, yaml
g = yaml.safe_load(open(sys.argv[1]))["general"]
print(g["workload_type"], g["workload"])
' "$RECIPE_DIR/metadata.yaml"
)
: "${MODEL_NAME:?could not read general.workload from $RECIPE_DIR/metadata.yaml}"

LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL is a required variable (e.g. /mnt/nvme/llmb)}
LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
SKIP_PULL=${SKIP_PULL:-false}
SKIP_CLONE=${SKIP_CLONE:-false}
PREFETCH_HF=${PREFETCH_HF:-false}
MODEL_SIZE=${MODEL_SIZE:-8b}
MODEL_SIZE=${MODEL_SIZE,,}

# The repo to prefetch comes from the recipe's declared downloads, matched on
# the size token (e.g. "-30b" matches Qwen/Qwen3-30B-A3B), so this works for any
# recipe instead of hardcoding one family's repo names.
HF_REPO=$(
    python3 -c '
import sys, yaml
meta = yaml.safe_load(open(sys.argv[1]))
size = sys.argv[2].lower()
repos = [d["repo_id"] for d in meta.get("downloads", {}).get("huggingface", [])]
match = [r for r in repos if f"-{size}" in r.lower()]
print(match[0] if match else (repos[0] if len(repos) == 1 else ""))
' "$RECIPE_DIR/metadata.yaml" "$MODEL_SIZE"
)
HF_REPO=${HF_REPO_OVERRIDE:-$HF_REPO}

export HF_HOME=${HF_HOME:-$LLMB_INSTALL/.cache/huggingface}
export NEMO_HOME=${NEMO_HOME:-$LLMB_INSTALL/.cache/nemo}

# --- Read the pins out of the recipe so this script can never drift from it ---
read -r MBRIDGE_URL MBRIDGE_COMMIT < <(
    python3 - "$RECIPE_DIR/metadata.yaml" <<'PY'
import sys, yaml
meta = yaml.safe_load(open(sys.argv[1]))
repo = meta["repositories"]["megatron_bridge"]
print(repo["url"], repo["commit"])
PY
)

FW_VERSION=$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?FW_VERSION=\(.*\)$/\2/p' "$RECIPE_DIR/launch.sh" | head -1)
if [[ -z $FW_VERSION ]]; then
    echo "error: could not read FW_VERSION from $RECIPE_DIR/launch.sh" >&2
    exit 1
fi
IMAGE=${IMAGE:-nvcr.io/nvidia/nemo:$FW_VERSION}

echo "==================================================================="
echo " Single-node Docker setup for ${WORKLOAD_TYPE}_${MODEL_NAME}"
echo "==================================================================="
echo " LLMB_INSTALL:    $LLMB_INSTALL"
echo " LLMB_WORKLOAD:   $LLMB_WORKLOAD"
echo " Megatron-Bridge: $MBRIDGE_URL @ $MBRIDGE_COMMIT"
echo " Image:           $IMAGE"
echo " HF_HOME:         $HF_HOME"
echo " NEMO_HOME:       $NEMO_HOME"
echo "==================================================================="

mkdir -p "$LLMB_WORKLOAD/experiments" "$HF_HOME" "$NEMO_HOME"

# --- 1. Megatron-Bridge checkout ---------------------------------------------
# Only scripts/performance/ is used at run time; the `megatron.bridge` library
# itself comes from the container, exactly as on the Slurm path.
MBRIDGE_DIR=$LLMB_WORKLOAD/Megatron-Bridge
if [[ $SKIP_CLONE == true ]]; then
    echo "[1/3] skipping clone (SKIP_CLONE=true)"
elif [[ -d $MBRIDGE_DIR/.git ]]; then
    current=$(git -C "$MBRIDGE_DIR" rev-parse HEAD)
    if [[ $current == "$MBRIDGE_COMMIT" ]]; then
        echo "[1/3] Megatron-Bridge already at $MBRIDGE_COMMIT"
    else
        echo "[1/3] fetching $MBRIDGE_COMMIT into existing checkout"
        git -C "$MBRIDGE_DIR" fetch --quiet origin "$MBRIDGE_COMMIT"
        git -C "$MBRIDGE_DIR" checkout --quiet "$MBRIDGE_COMMIT"
    fi
else
    echo "[1/3] cloning Megatron-Bridge"
    git clone --quiet "$MBRIDGE_URL" "$MBRIDGE_DIR"
    git -C "$MBRIDGE_DIR" checkout --quiet "$MBRIDGE_COMMIT"
fi
git -C "$MBRIDGE_DIR" --no-pager log --oneline -1

if [[ ! -f $MBRIDGE_DIR/scripts/performance/run_script.py ]]; then
    echo "error: $MBRIDGE_DIR/scripts/performance/run_script.py missing after checkout" >&2
    exit 1
fi

# --- 2. Container image -------------------------------------------------------
if [[ $SKIP_PULL == true ]]; then
    echo "[2/3] skipping pull (SKIP_PULL=true)"
elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[2/3] image already present: $IMAGE"
else
    echo "[2/3] pulling $IMAGE (~18 GB compressed; needs 'docker login nvcr.io')"
    docker pull "$IMAGE"
fi

# --- 3. Caches ----------------------------------------------------------------
if [[ $PREFETCH_HF == true ]]; then
    if [[ -z ${HF_TOKEN:-} ]]; then
        echo "error: PREFETCH_HF=true requires HF_TOKEN" >&2
        exit 1
    fi
    PROXY_ENV=()
    for var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
        [[ -n ${!var:-} ]] && PROXY_ENV+=(-e "$var")
    done
    echo "[3/3] prefetching $HF_REPO config into $HF_HOME"
    docker run --rm \
        -e HF_TOKEN \
        -e HF_HOME="$HF_HOME" \
        ${PROXY_ENV[@]+"${PROXY_ENV[@]}"} \
        -v "$HF_HOME:$HF_HOME" \
        "$IMAGE" \
        hf download "$HF_REPO" config.json generation_config.json
else
    echo "[3/3] cache dirs created; tokenizer will be fetched on first run"
fi

echo
# Prefer the recipe's own entry point if it has one, so the hint matches how
# the user actually invoked this rather than naming the shared implementation.
ENTRY_DIR=$SCRIPT_DIR
[[ -x $RECIPE_DIR/single_node_docker/launch_local.sh ]] && ENTRY_DIR=$RECIPE_DIR/single_node_docker

echo "Setup complete. Next:"
echo "  export LLMB_INSTALL=$LLMB_INSTALL"
echo "  export HF_TOKEN=\$(cat /path/to/hf-token)"
echo "  cd $ENTRY_DIR && JOB_TOTAL_GPUS=<n> GPU_TYPE=<type> MODEL_SIZE=$MODEL_SIZE ./launch_local.sh"
