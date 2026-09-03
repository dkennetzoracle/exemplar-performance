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
# Host prep for the perf matrix. Run once per node, ON the GPU node.
#
# Two things bite here and both cost a wasted image pull if missed:
#
#  1. Docker 25+ keeps images in *containerd's* root, not Docker's data-root.
#     Setting "data-root" in daemon.json looks like it works -- `docker info`
#     dutifully reports the new path -- but images never go there. Observed on
#     a Docker 29.6 node: /mnt/localdisk/docker held 4.2 MB while
#     /var/lib/containerd held 70 GB, and the second image pull died with
#     "no space left on device" writing to
#     /var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/.
#     The knob that actually moves images is `root` in
#     /etc/containerd/config.toml.
#
#  2. A training process with ~280 GB of device memory mapped writes a core
#     dump of roughly that size. One failed 4-rank run put 4 x 17 GB into
#     /var/lib/apport/coredump and filled the root filesystem. Bring-up means
#     crashes, so cores are disabled for containers.
#
# Usage:
#   IMAGE_STORE=/mnt/localdisk ./node_prep.sh
#
# Env:
#   IMAGE_STORE   parent dir for the relocated containerd root (required-ish;
#                 defaults to /mnt/localdisk, the local NVMe on these nodes)

set -euo pipefail

IMAGE_STORE=${IMAGE_STORE:-/mnt/localdisk}
TARGET=$IMAGE_STORE/containerd

echo "=== host: $(hostname) ==="
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
    --format=csv || echo "WARNING: nvidia-smi failed"
echo
echo "--- docker / toolkit ---"
docker --version
nvidia-container-cli --version 2>&1 | head -2 || echo "WARNING: nvidia-container-cli missing"

# overlayfs on xfs requires ftype=1; a mkfs.xfs without it cannot back images.
if findmnt -no FSTYPE "$IMAGE_STORE" 2>/dev/null | grep -q xfs; then
    echo "--- xfs ftype (overlayfs needs ftype=1) ---"
    xfs_info "$IMAGE_STORE" | grep -o 'ftype=[01]' || true
fi

current_root=$(containerd config dump 2>/dev/null | sed -n "s/^root = '\(.*\)'/\1/p")
if [[ $current_root != "$TARGET" ]]; then
    echo "--- relocating containerd root: $current_root -> $TARGET ---"
    sudo mkdir -p /etc/containerd "$TARGET"
    # This file merges over defaults, so one key is enough.
    printf 'version = 3\n\nroot = "%s"\n' "$TARGET" | sudo tee /etc/containerd/config.toml
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
    "default-ulimits": { "core": { "Name": "core", "Hard": 0, "Soft": 0 } }
}
JSON
    sudo systemctl stop docker.socket docker containerd
    # Preserve any images already pulled on this node.
    sudo rsync -aHAX --numeric-ids /var/lib/containerd/ "$TARGET"/
    sudo systemctl start containerd docker
    sleep 5
else
    echo "--- containerd root already at $TARGET ---"
fi

echo
echo "=== verify ==="
containerd config dump | grep -E '^root|^state'
docker info -f 'DockerRootDir={{.DockerRootDir}} StorageDriver={{.Driver}}'
df -hT / "$IMAGE_STORE" | grep -v ^Filesystem
echo "--- core ulimit inside a container (expect 0) ---"
docker run --rm busybox sh -c 'ulimit -c' 2>/dev/null \
    || echo "(no busybox cached; re-check after setup_images.sh)"
