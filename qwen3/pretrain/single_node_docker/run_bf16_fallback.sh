#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Entry point for running this recipe on one node without Slurm.
#
# The implementation is recipe-agnostic and lives once, in the llama3.1 recipe's
# single_node_docker/ directory; it selects a recipe via RECIPE_DIR. This
# wrapper just points it at qwen3/pretrain so the tooling is discoverable from
# the recipe it applies to, rather than having to be invoked from another
# recipe's directory.
#
# See ../../../llama3.1/pretrain/megatron_bridge/single_node_docker/QUICKSTART.md
# for setup, and ./README.md for what is specific to Qwen3.

set -eu -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMPL_DIR=$SCRIPT_DIR/../../../llama3.1/pretrain/megatron_bridge/single_node_docker

[[ -x $IMPL_DIR/run_bf16_fallback.sh ]] || {
    echo "error: $IMPL_DIR/run_bf16_fallback.sh not found or not executable" >&2
    exit 1
}

export RECIPE_DIR=${RECIPE_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
# Qwen3's smallest single-node-plausible size; 235b needs far more than one node.
export MODEL_SIZE=${MODEL_SIZE:-30b}
export WORKLOAD=${WORKLOAD:-qwen3-30b}

exec "$IMPL_DIR/run_bf16_fallback.sh" "$@"
