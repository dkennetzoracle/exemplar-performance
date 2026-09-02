# Quickstart: run this recipe on one node, no Slurm

Every command below was run on the machine this was developed against
(4 × compute-capability 10.7, aarch64, Ubuntu 24.04, Docker 29). Steps 5 and 6
are the ones that produce a training run; steps 1–4 are host setup you do once.

If you are on a GPU the container already supports, you can skip to
[Step 6a](#6a-normal-launch) and ignore the fallback entirely.

---

## 1. Host prerequisites

```bash
docker --version                 # 25+ so the containerd image store is used
nvidia-smi                       # driver present
nvidia-container-cli --version   # NVIDIA Container Toolkit installed
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
python3 --version                # 3.12 on the host, for the resolver
```

The launcher needs `docker`, `python3` and `nvidia-smi` on the host, plus
PyYAML (`python3 -c 'import yaml'`) for reading the recipe's `metadata.yaml`.

### Behind a proxy

Two separate things need it. The **Docker daemon** pulls images, so it needs its
own configuration — your shell's `https_proxy` does not reach it:

```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
    "runtimes": { "nvidia": { "args": [], "path": "nvidia-container-runtime" } },
    "proxies": {
        "http-proxy":  "http://your-proxy:80",
        "https-proxy": "http://your-proxy:80",
        "no-proxy":    "localhost,127.0.0.1,::1"
    },
    "default-ulimits": { "core": { "Name": "core", "Hard": 0, "Soft": 0 } }
}
EOF
sudo systemctl restart docker
docker info | grep -i proxy          # verify
docker run --rm <image> bash -c 'ulimit -c'   # expect 0
```

Keep any `runtimes` block you already have. The `default-ulimits` entry is not
about proxies — see [Step 3](#3-stop-crashing-ranks-filling-the-root-disk).

Then export the proxy in your shell too, so `launch_local.sh` can forward it
into the container for HuggingFace access:

```bash
export https_proxy=http://your-proxy:80
export http_proxy=$https_proxy
export no_proxy=localhost,127.0.0.1,::1
```

---

## 2. Put the image store somewhere with room

The NeMo container is ~19 GB compressed and ~60–65 GB unpacked. **Check where
images actually live before planning anything** — with Docker 25+ and the
containerd image store they are in *containerd's* root, so moving Docker's
`data-root` moves nothing:

```bash
du -sh /var/lib/docker /var/lib/containerd
containerd config dump | grep -E '^root|^state'
```

To relocate (example target `/mnt/nvme`):

```bash
# 2a. Make the mount permanent FIRST. If it is only mounted by hand, a reboot
#     leaves containerd with an empty store.
blkid /dev/md0                                    # get the UUID
sudo cp /etc/fstab /etc/fstab.bak
echo 'UUID=<uuid> /mnt/nvme ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo systemctl daemon-reload && findmnt --verify --fstab

# 2b. Point containerd at it (this file merges over defaults, so one key is enough)
sudo mkdir -p /etc/containerd
printf 'version = 3\n\nroot = "/mnt/nvme/containerd"\n' | sudo tee /etc/containerd/config.toml

# 2c. Move the data
sudo systemctl stop docker.socket docker containerd
sudo mkdir -p /mnt/nvme/containerd
sudo rsync -aHAX --numeric-ids /var/lib/containerd/ /mnt/nvme/containerd/
sudo systemctl start containerd docker

# 2d. VERIFY before deleting anything
containerd config dump | grep '^root'             # expect /mnt/nvme/containerd
docker images                                     # expect your images intact
docker run --rm --gpus '"device=0"' <image> nvidia-smi -L
sudo mv /var/lib/containerd /var/lib/containerd.old   # rename, do not rm yet
docker images && sudo rm -rf /var/lib/containerd.old  # only once confirmed
```

---

## 3. Stop crashing ranks filling the root disk

A training process with ~280 GB of device memory mapped writes a core dump of
roughly that size. One failed 4-rank run put 4 × 17 GB into
`/var/lib/apport/coredump` and filled the root filesystem. Bring-up means
crashes, so turn cores off for containers — that is the `default-ulimits` block
in [Step 1](#behind-a-proxy). To clear existing dumps:

```bash
du -sh /var/lib/apport/coredump
sudo rm -f /var/lib/apport/coredump/core.*
```

---

## 4. Credentials

**NGC**, for pulling the container. Create an API key at
<https://ngc.nvidia.com/setup/api-key>, then (username is the literal string
`$oauthtoken`):

```bash
printf '%s' '<your-ngc-key>' | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Piping from a file keeps the key out of your shell history:

```bash
tr -d '[:space:]' < ~/.ngc | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

**HuggingFace**, because the recipe builds its model config from a gated repo at
startup (config only — no weights are downloaded). Which repo depends on the
config you run, and this is the most common setup failure:

| `MODEL_RECIPE_NAME` | reads | gated behind |
| --- | --- | --- |
| `llama3_8b` (default for `MODEL_SIZE=8b`) | `meta-llama/Meta-Llama-3-8B` | Llama **3** |
| `llama3_70b` | `meta-llama/Meta-Llama-3-70B` | Llama **3** |
| `llama31_8b` | `meta-llama/Meta-Llama-3.1-8B` | Llama **3.1** |
| `llama31_405b` | `meta-llama/Meta-Llama-3.1-405B` | Llama **3.1** |

Llama 3 and Llama 3.1 are approved **separately**. Check before launching —
`200` or `307` means access, `403` means approval still pending:

```bash
export HF_TOKEN=$(tr -d '[:space:]' < ~/.hf)
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json
```

A `403` here is what produces the confusing
`filelock._error.Timeout ... .megatron_config_lock` failure later: all ranks
serialize on one lock while the first rank's request fails.

Keep these files out of git:

```bash
printf '\n.ngc\n.hf\n' >> .gitignore
```

---

## 5. Install the workload

```bash
export LLMB_INSTALL=/mnt/nvme/llmb
cd llama3.1/pretrain/megatron_bridge/single_node_docker
./setup_local.sh
```

Clones Megatron-Bridge at the commit pinned in `../metadata.yaml`, pulls the
NeMo container named by `FW_VERSION` in `../launch.sh`, and creates the cache
directories. Both pins are read out of the recipe, so this cannot drift from
the Slurm path.

Optionally pre-download the config so runs need no network:

```bash
PREFETCH_HF=true SKIP_PULL=true SKIP_CLONE=true ./setup_local.sh
```

### Check the container supports your GPU

Worth a minute before a 50-step run:

```bash
./check_arch_support.sh nvcr.io/nvidia/nemo:26.06.01
```

`bf16 OK` with `fp8_*`/`nvfp4`/`torch.compile` all `FAIL` and
`ptxas accepts sm_<cc>a: MISSING` means the container has no kernels for your
GPU. Full per-image evidence lives in [`arch_support/`](arch_support/).

---

## 6. Launch

### 6a. Normal launch

Use this when `check_arch_support.sh` is clean:

```bash
export LLMB_INSTALL=/mnt/nvme/llmb
export HF_TOKEN=$(tr -d '[:space:]' < ~/.hf)

JOB_TOTAL_GPUS=4 GPU_TYPE=vr200 MODEL_SIZE=8b ./launch_local.sh
```

Sanity-check the plumbing without touching a GPU first:

```bash
DRYRUN=true    JOB_TOTAL_GPUS=4 GPU_TYPE=vr200 MODEL_SIZE=8b ./launch_local.sh
PRINT_ONLY=true JOB_TOTAL_GPUS=4 GPU_TYPE=vr200 MODEL_SIZE=8b ./launch_local.sh
```

### 6b. Fallback launch on an unsupported arch

```bash
export LLMB_INSTALL=/mnt/nvme/llmb
export HF_TOKEN=$(tr -d '[:space:]' < ~/.hf)

./run_bf16_fallback.sh
```

[`run_bf16_fallback.sh`](run_bf16_fallback.sh) wraps `launch_local.sh` with the
six workarounds and documents why each is needed. It is the exact configuration
that produced the numbers below. Expanded, it is:

```bash
COMPAT_SHIM=true \
EXTRA_ENV="TORCHDYNAMO_DISABLE=1 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=1" \
EXTRA_HYDRA_OVERRIDES="mixed_precision.fp4=null \
  mixed_precision.fp8_dot_product_attention=false \
  model.fp8_dot_product_attention=false \
  model.use_transformer_engine_op_fuser=false \
  model.cross_entropy_loss_fusion=false \
  model.recompute_granularity=full \
  model.recompute_method=uniform \
  model.recompute_num_layers=1" \
MBS=1 GBS=8 MAX_STEPS=50 \
MODEL_RECIPE_NAME=llama31_8b GPU_TYPE=gb200 DTYPE=nvfp4 CONFIG_VARIANT=v1 \
JOB_TOTAL_GPUS=4 MODEL_SIZE=8b ./launch_local.sh
```

> **This is not a benchmark configuration.** bf16 instead of nvfp4/fp8, eager
> instead of fused activations, unfused instead of cuDNN attention, `MBS=1`,
> full activation recompute, and 4 GPUs against a validated minimum of 8. All
> six are slowdowns. `DTYPE=nvfp4` selects the *preset*; the precision is
> overridden to bf16, so the experiment name says `nvfp4` while the run is
> bf16. Do not compare the result to published 8B figures.

---

## 7. Read the results

`launch_local.sh` prints them when the run finishes, averaged over iterations
35–44 by the repo's own parser — same window and same NaN-grad-norm rejection
as the Slurm path:

```
===================================================================
 Results
===================================================================
 Averaging window:   iterations 35-44
 Samples:            10
 s/iter:             2.504 (std 0.000)
 TFLOPS/GPU:         336.82 (std 0.04)
===================================================================
```

To re-read a finished run:

```bash
python3 parse_results_local.py \
  $LLMB_INSTALL/workloads/pretrain_llama3.1/experiments/<exp>/<exp>_<ts>/<exp>/log-<exp>.out
```

Each run directory holds `log-<exp>.out`, `env.list` (the resolved container
environment, useful for confirming an override landed) and
`configs/ConfigContainer.yaml` (the fully resolved config).

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `cannot set both Count and DeviceIDs on device request` | `--gpus` needs its device list literally quoted. The launcher already does this. |
| Every rank exits 1, `set_mempolicy: Operation not permitted` | `numactl` needs `CAP_SYS_NICE`. The launcher passes `--cap-add=SYS_NICE`. |
| `filelock._error.Timeout ... .megatron_config_lock` | Gated HF repo returned 403. See [Step 4](#4-credentials). |
| `PTXAS error: Internal Triton PTX codegen error`, `sm_XXXa is not a recognized processor` | Container CUDA is older than the GPU. `EXTRA_ENV="TORCHDYNAMO_DISABLE=1"`, or [6b](#6b-fallback-launch-on-an-unsupported-arch). |
| `no kernel image is available for execution on the device` | TE has no cubin for the arch. bf16 only; see [6b](#6b-fallback-launch-on-an-unsupported-arch). |
| `cuDNN Error: ... No valid execution plans built` | No fused-attention plan for the arch. `NVTE_UNFUSED_ATTN=1`. |
| `LLVM ERROR: Cannot select ... shfl.sync.bfly.i32` | TE fused cross-entropy is Triton. `model.cross_entropy_loss_fusion=false`. |
| `torch.OutOfMemoryError` right after the first step | Unfused attention materializes the score matrix. `MBS=1` plus full recompute. |
| `TypeError: unsupported format string passed to Tensor.__format__` | Upstream `grad_norm` tensor-vs-float bug. `COMPAT_SHIM=true`. |
| Root filesystem fills during a crash loop | Core dumps. See [Step 3](#3-stop-crashing-ranks-filling-the-root-disk). |
| `error: container image '...' not present locally` | `./setup_local.sh`, or `docker pull` it. |

See the [README](README.md) for what this replaces on the Slurm side and the
detail behind each failure mode.
