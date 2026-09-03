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
# Image + checkout + parser setup for the perf matrix. Run once per node,
# ON the GPU node, after node_prep.sh.
#
# Does four things:
#   1. logs in to nvcr.io using the key file (never the shell history)
#   2. reports HuggingFace access for each gated repo the recipes read
#   3. runs setup_local.sh for both recipes (llama3.1 + qwen3), for each image
#   4. builds a venv with the repo parser's deps, so launch_local.sh can print
#      its own Results block instead of erroring on a missing 'typer'
#
# Usage:
#   ./setup_images.sh                     # recipe-pinned image only
#   IMAGES="26.06.01 26.08.00" ./setup_images.sh
#
# Env:
#   IMAGES         space-separated nemo tags to pull (default: the recipe pin)
#   LLMB_INSTALL   install root (default /mnt/localdisk/llmb)

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

# The recipe's own pin, so the default can never drift from the Slurm path.
FW_VERSION=$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?FW_VERSION=\(.*\)$/\2/p' \
    "$SND_DIR/../launch.sh" | head -1)
IMAGES=${IMAGES:-$FW_VERSION}

echo "=== host: $(hostname) ==="
echo " recipe pin:   $FW_VERSION"
echo " images:       $IMAGES"
echo " LLMB_INSTALL: $LLMB_INSTALL"
echo

if [[ -r $NGC_TOKEN_FILE ]]; then
    tr -d '[:space:]' < "$NGC_TOKEN_FILE" | docker login nvcr.io -u '$oauthtoken' --password-stdin
else
    echo "WARNING: $NGC_TOKEN_FILE missing; assuming you are already logged in to nvcr.io" >&2
fi

# --- HuggingFace access ------------------------------------------------------
# 200/307 = access, 403 = approval still pending. A 403 is what later shows up
# as the confusing 'filelock._error.Timeout ... .megatron_config_lock': every
# rank serialises on one lock while rank 0's request fails.
# Llama 3 and Llama 3.1 are approved SEPARATELY -- llama3_8b (the MODEL_SIZE=8b
# default) reads Meta-Llama-3-8B, llama31_8b reads Meta-Llama-3.1-8B.
if [[ -n ${HF_TOKEN:-} ]]; then
    echo
    echo "--- HuggingFace access (403 = not approved yet) ---"
    for repo in meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3.1-8B Qwen/Qwen3-30B-A3B; do
        code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" \
            "https://huggingface.co/$repo/resolve/main/config.json" || echo "ERR")
        printf '  %-34s %s\n' "$repo" "$code"
    done
else
    echo "WARNING: no HF_TOKEN; put a token in $HF_TOKEN_FILE" >&2
fi

# --- Images + Megatron-Bridge checkouts --------------------------------------
first=true
for tag in $IMAGES; do
    echo
    echo "=== setup for nemo:$tag ==="
    # Clone only once; the checkout is shared and bind-mounted read-only.
    if $first; then
        IMAGE="nvcr.io/nvidia/nemo:$tag" "$SND_DIR/setup_local.sh"
        IMAGE="nvcr.io/nvidia/nemo:$tag" SKIP_PULL=true SKIP_CLONE=true \
            RECIPE_DIR="$REPO_ROOT/qwen3/pretrain" MODEL_SIZE=30b "$SND_DIR/setup_local.sh"
        first=false
    else
        IMAGE="nvcr.io/nvidia/nemo:$tag" SKIP_CLONE=true "$SND_DIR/setup_local.sh"
    fi
done

# --- Parser venv -------------------------------------------------------------
# parse_results_local.py imports llmb_run.pretrain_log_parser from cli/llmb-run,
# which needs typer et al. Without these, every run trains fine and then prints
# "could not import llmb_run.pretrain_log_parser (No module named 'typer')".
if [[ ! -x $PARSER_VENV/bin/python ]]; then
    echo
    echo "=== building parser venv at $PARSER_VENV ==="
    python3 -m venv --system-site-packages "$PARSER_VENV"
    "$PARSER_VENV/bin/pip" install -q \
        'typer~=0.25' 'pydantic>=2.0,<3' 'rich>=13.8,<16' 'zstandard~=0.23'
fi
"$PARSER_VENV/bin/python" -c 'import typer, pydantic, rich, zstandard; print("parser deps OK")'

echo
echo "=== done ==="
docker images | grep -E 'REPOSITORY|nemo'
df -hT / "${IMAGE_STORE:-/mnt/localdisk}" | grep -v ^Filesystem
echo
echo "Next: ./run_matrix.sh reference    (then native / portable)"
