# Single-node Docker launch (no Slurm)

The recipe in the parent directory only supports Slurm: [`../launch.sh`](../launch.sh)
bails out with `Only SLURM is supported for this workload`, then hands the job to
Nemo-Run's `SlurmExecutor`, which writes an `sbatch` script and fans out ranks
with `srun`.

This directory runs the same workload on a single node with nothing but Docker
and `torchrun`.

## What this replaces, and what it does not

The rank-local entrypoint is **unchanged**. Both paths run the same
`scripts/performance/run_script.py` from the Megatron-Bridge commit pinned in
[`../metadata.yaml`](../metadata.yaml), bind-mounted at its host path, against
the `megatron.bridge` library shipped inside the NeMo container. Only the
orchestration around it is reimplemented:

| Slurm path | This path |
| --- | --- |
| `sbatch` + `srun --mpi=pmix` | `docker run` + `torchrun --nnodes=1` |
| Pyxis/Enroot `--container-image` | `docker run <image>` |
| `--container-mounts` | `docker run -v` |
| `numactl --cpunodebind=$((SLURM_LOCALID/N))` prepended to each task | [`rank_wrapper_local.sh`](rank_wrapper_local.sh), dividing `LOCAL_RANK` |
| `PERF_ENV_VARS` + `PerfEnvPlugin` mutating `executor.env_vars` | [`perf_env_local.py`](perf_env_local.py) → `docker --env-file` |
| `NsysPlugin` wrapping each task in `nsys profile` | `NSYS_ENABLE=1` in the rank wrapper |
| `llmb-run jobs` (refreshes Slurm state) | [`parse_results_local.py`](parse_results_local.py) |

[`check_arch_support.sh`](check_arch_support.sh) is an extra, not a port of
anything: it answers "can this container train on this GPU?" in about a minute,
which is worth knowing before committing to a 50-step run on new silicon.

`perf_env_local.py` reimplements the environment *rules* only. The *data* those
rules branch on — parallelism, batch sizes, CUDA-graph mode, MoE backend, FSDP
and NCCL-UB flags — is imported live from the pinned checkout via
`utils.utils.get_workload_base_config`, and the experiment name comes from
upstream's own `get_exp_name_config`. Upstream config changes are therefore
picked up automatically; only a change to the env rules themselves would need
mirroring. The mirrored rules are cross-referenced to their source functions in
comments and were checked against commit `b50da4c7404caa41793e74ac40d18798844c7b67`.

**Not ported:** multi-node, Nemo-Run experiment bookkeeping and retries, the
Kubeflow/CSP executors, and `llmb-run`'s job database.

## Verified state

Exercised on a single node of 4 × compute-capability 10.7 GPUs (aarch64, 4 GPUs
across 2 NUMA nodes) with `nvcr.io/nvidia/nemo:26.06.01`. A full 50-step run
completes and the repo's own parser reports metrics over the standard window:

```
 Averaging window:   iterations 35-44
 Samples:            10
 s/iter:             2.504 (std 0.000)
 TFLOPS/GPU:         336.82 (std 0.04)
```

Command (`llama31_8b` because this token has Llama 3.1 but not Llama 3 access;
see [HuggingFace access](#huggingface-access-llama-3-vs-llama-31)):

```bash
COMPAT_SHIM=true \
EXTRA_ENV="TORCHDYNAMO_DISABLE=1 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=1" \
EXTRA_HYDRA_OVERRIDES="mixed_precision.fp4=null \
  mixed_precision.fp8_dot_product_attention=false \
  model.fp8_dot_product_attention=false \
  model.use_transformer_engine_op_fuser=false \
  model.cross_entropy_loss_fusion=false \
  model.recompute_granularity=full model.recompute_method=uniform \
  model.recompute_num_layers=1" \
MBS=1 GBS=8 MAX_STEPS=50 MODEL_RECIPE_NAME=llama31_8b GPU_TYPE=gb200 \
  DTYPE=nvfp4 CONFIG_VARIANT=v1 JOB_TOTAL_GPUS=4 MODEL_SIZE=8b ./launch_local.sh
```

> **That number is a bring-up baseline, not a benchmark result.** Six deviations
> stack up, every one of them a slowdown: bf16 instead of nvfp4/fp8 (no kernels
> exist for this arch), eager instead of fused activations, unfused instead of
> cuDNN attention, `MBS=1`, full activation recompute, and 4 GPUs instead of the
> validated minimum of 8. `DTYPE=nvfp4` in the command selects the *preset*; the
> precision is overridden to bf16, so the experiment name says `nvfp4` while the
> run is bf16. Do not compare it to the published 8B figures.

What this does establish is that the port itself is complete and correct: image
pull, container launch, 4 ranks under `torchrun`, per-rank NUMA binding,
resolved perf environment and Hydra overrides, gated-HF config fetch, model
init, DDP bucketing, distributed optimizer, mock dataset build, 50 training
iterations with a falling loss, post-training eval, and metric parsing.

Getting a *comparable* number needs a container with kernels for the arch — see
[Triton / `ptxas` vs. new silicon](#triton--ptxas-vs-new-silicon).

## Setup

```bash
docker login nvcr.io          # username: $oauthtoken, password: your NGC API key
export LLMB_INSTALL=/mnt/nvme/llmb
./setup_local.sh
```

`setup_local.sh` clones Megatron-Bridge at the pinned commit, pulls the NeMo
container named by `FW_VERSION` in `../launch.sh` (~18 GB compressed, ~60 GB on
disk), and creates the cache directories. Both the commit and the image tag are
read out of the recipe, so this cannot drift from `../launch.sh`.

Behind a proxy, export `https_proxy`/`http_proxy` before running: the Docker
*daemon* needs its own proxy config for `docker pull` (`/etc/docker/daemon.json`
→ `"proxies"`, then restart Docker), while `launch_local.sh` forwards the proxy
variables into the container for HuggingFace access.

### Host storage notes

Two things bite on a machine with a small root filesystem:

**Images may not live under `/var/lib/docker`.** Docker 25+ with the containerd
image store keeps layers in *containerd's* root, so moving Docker's `data-root`
moves nothing. Check before planning a migration:

```bash
du -sh /var/lib/docker /var/lib/containerd
containerd config dump | grep -E '^root|^state'
```

To relocate, set containerd's `root` (not Docker's `data-root`) — containerd
merges `/etc/containerd/config.toml` over its defaults, so only the changed key
is needed:

```toml
version = 3
root = "/mnt/nvme/containerd"
```

Stop `docker.socket docker containerd`, `rsync -aHAX` the old root across, start
them again, and confirm `docker images` is intact before deleting the original.
Give the target an `/etc/fstab` entry with `nofail` first — if it is only
manually mounted, a reboot leaves containerd with an empty store.

**Crashing ranks produce enormous core dumps.** A training process with ~280 GB
of device memory mapped dumps a core of roughly that size; one failed 4-rank run
wrote 4 × 17 GB into `/var/lib/apport/coredump` and filled the root filesystem.
During bring-up, where crashes are expected, turn cores off for containers in
`/etc/docker/daemon.json`:

```json
"default-ulimits": { "core": { "Name": "core", "Hard": 0, "Soft": 0 } }
```

Verify with `docker run --rm <image> bash -c 'ulimit -c'` → `0`.

## Run

```bash
export LLMB_INSTALL=/mnt/nvme/llmb
export HF_TOKEN=<your token>
JOB_TOTAL_GPUS=4 GPU_TYPE=vr200 MODEL_SIZE=8b ./launch_local.sh
```

The env-var interface matches `../launch.sh` (`JOB_TOTAL_GPUS`, `GPU_TYPE`,
`MODEL_SIZE`, `DTYPE`, `FP8_RECIPE`, `CONFIG_VARIANT`, `MAX_STEPS`,
`ENABLE_PROFILE`, `ENABLE_PYTORCH_PROFILE`, `ENABLE_GPU_METRICS`,
`ENABLE_CHECKPOINT`, `CHECKPOINT_DIR`, `LOAD_CHECKPOINT_PATH`,
`TP`/`PP`/`CP`/`VP`/`MBS`/`GBS`). Differences:

| Variable | Here | `../launch.sh` |
| --- | --- | --- |
| `MODEL_SIZE` | defaults to `8b` | defaults to `405b` (needs ≥ 256 GPUs) |
| `MODEL_RECIPE_NAME` | override the Megatron-Bridge config to load | derived from `MODEL_SIZE` only |
| `OFFLINE` | `true` sets `HF_HUB_OFFLINE=1` | always passes `--offline` |
| `DRYRUN` | `true` dumps `ConfigContainer.yaml` and exits | n/a |
| `PRINT_ONLY` | `true` prints the assembled command and exits | n/a |
| `MASTER_PORT` | `torchrun` rendezvous port (default 29500) | n/a |
| `EXTRA_ENV` | extra `KEY=VALUE` container env, appended to `env.list` | n/a |
| `EXTRA_HYDRA_OVERRIDES` | extra Hydra overrides for `run_script.py` | n/a |
| `COMPAT_SHIM` | `true` applies library-bug workarounds (see below) | n/a |
| `CG_IMPL` / `CG_SCOPE` | override CUDA-graph mode (`CG_IMPL=none` disables) | preset only |
| `SBATCH_*`, `ADDITIONAL_SLURM_PARAMS`, `TIME_LIMIT` | ignored | required / used |

Results land in the same layout the parent README documents, under
`$LLMB_INSTALL/workloads/pretrain_llama3.1/experiments/<exp>/<exp>_<ts>/<exp>/`,
with `log-<exp>.out`, `env.list` (the resolved container environment) and
`configs/ConfigContainer.yaml`. `launch_local.sh` prints `s/iter` and
`TFLOPS/GPU` when the run finishes, averaged over iterations 35–44 by the
repo's own parser — the same window and the same NaN-grad-norm rejection the
Slurm path applies.

Start with `DRYRUN=true` on a new machine: it exercises the container, the
mounts and the whole config resolution without touching a GPU.

## GPU_TYPE on Vera Rubin

`GPU_TYPE` only selects a config preset; it does not have to name the silicon.
The pinned commit already knows `vr200` (4 GPUs/node, `sm100_or_newer`,
NUMA divisor 2), and for `llama3_8b` the `vr200` presets are defined as aliases
of the `gb200` ones, so the two are interchangeable for this workload.

`../metadata.yaml` has no `vr200` entry, so the launcher prints a note that the
scale is unvalidated and proceeds.

## Running below the documented scale

The parent README documents 8B at 8–128 GPUs, and `../metadata.yaml`'s smallest
listed scale is 8. Four GPUs still fits: 8B is DP-only (TP=PP=CP=1), so it
shards cleanly, and global batch size follows the recipe's own weak scaling
(`gbs_scaling_factor × num_gpus`).

The launcher prints a note whenever `JOB_TOTAL_GPUS` is outside the validated
list. **Results from an unvalidated scale are not comparable to published
numbers** — weak scaling changes the global batch size, so `s/iter` is not
comparable across scales even though `TFLOPS/GPU` is roughly so.

If weak scaling floors the global batch size below 1 — presets tuned for
hundreds of GPUs have `gbs_scaling_factor < 1` — the launcher stops and asks for
an explicit `GBS`.

## HuggingFace access: Llama 3 vs Llama 3.1

The recipe builds its model config from a gated HuggingFace repo at startup
(`AutoBridge.from_hf_pretrained(...)`, weights not loaded). Which repo depends
on the config, and this is the most common setup failure:

| `MODEL_RECIPE_NAME` | HF repo read | Gated behind |
| --- | --- | --- |
| `llama3_8b` (default for `MODEL_SIZE=8b`) | `meta-llama/Meta-Llama-3-8B` | Llama **3** |
| `llama3_70b` | `meta-llama/Meta-Llama-3-70B` | Llama **3** |
| `llama31_8b` | `meta-llama/Meta-Llama-3.1-8B` | Llama **3.1** |
| `llama31_405b` | `meta-llama/Meta-Llama-3.1-405B` | Llama **3.1** |

As the parent README notes, the 8B/70B configs intentionally reuse the
Megatron-Bridge `llama3` recipes, so they need Llama 3 access — approved
*separately* from Llama 3.1. A token with only Llama 3.1 access gets a `403` and
the run dies in `safe_load_config_with_retry` after four attempts, reported as a
file-lock timeout because all ranks serialize on one lock while the first rank's
request fails.

To check a token without launching anything:

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json
# 200 or 307 = access; 403 = approval still needed
```

With Llama 3.1 access only, run the genuine Llama 3.1 8B config instead:

```bash
MODEL_RECIPE_NAME=llama31_8b GPU_TYPE=gb200 DTYPE=nvfp4 CONFIG_VARIANT=v1 \
  JOB_TOTAL_GPUS=4 MODEL_SIZE=8b ./launch_local.sh
```

`llama31_8b` has `gb200`/`gb300` presets only (no `vr200`), and its `v1` preset
is the small-scale one (8 GPUs, GBS 16, MBS 2), which weak-scales to GBS 8 on
four GPUs. Its `fp8_cs` preset is tuned for 512 GPUs with TP=2/CP=4 and needs
`TP=1 CP=1 GBS=<n>` overrides to fit on one node.

Same layer count, width and head counts as Llama 3 8B, so step time is in the
same neighbourhood — but it is a **different model**
(`max_position_embeddings` 131072 and `llama3` rope scaling, factor 8, versus
8192 and no scaling), so its numbers are not a substitute for published Llama 3
8B results.

## Triton / `ptxas` vs. new silicon

If every rank dies during startup with

```
triton.runtime.errors.PTXASError: PTXAS error: Internal Triton PTX codegen error
'sm_XXXa' is not a recognized processor for this target (ignoring processor)
```

the container's CUDA toolkit is older than the GPU. Triton derives its target
from `torch.cuda.get_device_capability()` as `sm_<major><minor>a` and shells out
to the bundled `ptxas`; if that arch is not in `ptxas`'s allowed list, every
`torch.compile` region fails to build.

`./check_arch_support.sh <image>` reports this and everything below in one shot:

```
  compute_cap        sm_107
  ptxas_archs        sm_100 sm_100a sm_100f sm_103 sm_103a sm_103f sm_110 ...
  ptxas_target       MISSING sm_107a not in ptxas target list
  te_cubin_archs     sm_100 sm_100a sm_103a sm_120 sm_75 sm_80 sm_89 sm_90 sm_90a
  te_ptx_entries     0 (0 means no JIT fallback)
  precision_bf16     OK
  precision_fp8_cs   FAIL ...
  precision_nvfp4    FAIL ...hadamard_transform...
  torch_compile      FAIL ...
```

Note the `f` suffixes in the `ptxas` list: `sm_103f` is a *family* target, valid
on later `sm_10x` parts, whereas `sm_103a` is arch-exact. A TE built against
`sm_103f` would cover capability 10.7; the `sm_103a` build in 26.06.01 does not.
That distinction is the thing to look for when judging a newer container.

NeMo 26.06.01 ships CUDA 13.2, whose `ptxas` accepts `sm_100/103/110/120/121`
but not `sm_107` — so on a compute-capability 10.7 part, no Triton JIT can
build. Only `torch.compile` regions are affected; TransformerEngine, cuBLAS,
cuDNN and NCCL kernels are precompiled and run fine.

The workaround is to keep `torch.compile` out of the picture:

```bash
EXTRA_ENV="TORCHDYNAMO_DISABLE=1" ... ./launch_local.sh
```

Megatron's `jit_fuser` (used by `bias_activation_fusion` for fused bias+SwiGLU,
`bias_dropout_fusion`, and others) is `torch.compile`, so disabling Dynamo makes
those fall back to eager rather than failing. That costs some performance, so
**numbers measured this way are a lower bound** — note it alongside any result.
Disabling the individual fusions instead
(`EXTRA_HYDRA_OVERRIDES="model.bias_activation_fusion=false"`) is narrower but
has to be repeated for every fusion that trips the same wall. The real fix is a
container whose CUDA toolkit knows the target arch.

## Missing TransformerEngine kernel images

A second, independent symptom of the same container/silicon gap:

```
RuntimeError: .../transformer_engine/common/.../hadamard_transform.cu:599
  in function hadamard_transform_amax:
  CUDA Error: no kernel image is available for execution on the device
```

`libtransformer_engine.so` ships cubins for a fixed list of architectures and
**no PTX**, so there is no JIT fallback: if your arch is not in the list, those
kernels cannot run. Check coverage against your GPU:

```bash
docker run --rm <image> /usr/local/cuda/bin/cuobjdump --list-elf \
  /opt/venv/lib/python3.12/site-packages/transformer_engine/libtransformer_engine.so \
  | grep -oE 'sm_[0-9]+[af]?' | sort -u
docker run --rm <image> /usr/local/cuda/bin/cuobjdump --list-ptx \
  /opt/venv/lib/python3.12/site-packages/transformer_engine/libtransformer_engine.so | head
```

In NeMo 26.06.01 that list is `sm_75 sm_80 sm_89 sm_90 sm_90a sm_100 sm_100a
sm_103a sm_120`, with zero PTX entries. On a compute-capability 10.7 part every
quantized precision therefore fails — confirmed directly:

```python
# fp8_cs  -> fp8/quantize kernel: no kernel image available
# fp8_mx  -> device-side failure
# nvfp4   -> hadamard_transform: no kernel image available
# bf16    -> works
```

bf16 works because those paths land in cuBLAS/cuDNN, which are versioned with
the driver rather than baked into TE.

`llama3_8b` has bf16 presets for every GPU type, but reads the Llama 3-gated
repo. `llama31_8b` reads the Llama 3.1 repo but ships only `nvfp4` and `fp8_cs`
presets. To run bf16 off the Llama 3.1 config, take the `nvfp4` preset for its
parallelism and batch sizing and override the precision — `bf16_with_nvfp4_mixed`
differs from `bf16_mixed` in exactly `fp4`, `fp4_param` and `fp4_param_gather`,
and the preset already clears the latter two:

```bash
EXTRA_ENV="TORCHDYNAMO_DISABLE=1" \
EXTRA_HYDRA_OVERRIDES="mixed_precision.fp4=null \
  mixed_precision.fp8_dot_product_attention=false \
  model.fp8_dot_product_attention=false \
  model.use_transformer_engine_op_fuser=false" \
MODEL_RECIPE_NAME=llama31_8b GPU_TYPE=gb200 DTYPE=nvfp4 CONFIG_VARIANT=v1 \
  JOB_TOTAL_GPUS=4 MODEL_SIZE=8b ./launch_local.sh
```

The experiment name still says `nvfp4` because it is derived from
`--compute_dtype`, which selects the preset. **This is a bf16 run** — record it
as such. It is a bring-up path for new silicon, not a benchmark result.

### …and then attention

With quantization out of the way the next wall is TE's fused attention:

```
RuntimeError: .../fused_attn/fused_attn_f16_arbitrary_seqlen.cu:422
  cuDNN Error: [cudnn_frontend] Error: No valid execution plans built.
```

cuDNN has no execution plan for the arch. TE picks its attention backend from
`NVTE_FUSED_ATTN` / `NVTE_FLASH_ATTN` / `NVTE_UNFUSED_ATTN`, so force the
unfused one:

```bash
EXTRA_ENV="TORCHDYNAMO_DISABLE=1 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=1"
```

The unfused path materializes the full attention score matrix, which at
`seq_length=8192` costs `mbs × heads × 8192 × 8192 × 2 B` = 8 GiB per tensor at
`MBS=2`. That OOMs a 280 GB GPU (273 GB allocated), so it also needs a smaller
micro-batch and activation recompute:

```bash
MBS=1 GBS=8 \
EXTRA_HYDRA_OVERRIDES="... model.recompute_granularity=full \
  model.recompute_method=uniform model.recompute_num_layers=1"
```

### …and then it stops being worth it

That gets as far as `Starting training loop at iteration 0`, where a Triton
kernel invoked directly (not through `torch.compile`, which is already off)
aborts the process:

```
'sm_107a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32
```

This is an `LLVM ERROR` abort inside the compiler, not a Python exception, so
there is no flag left to turn off and nothing to catch.

**Conclusion: no publicly released container can train on a compute-capability
10.7 GPU.** Five independent subsystems have no code for the arch — `ptxas`,
TE's quantized kernels, cuDNN fused attention, and Triton both via Inductor and
directly — and each workaround only exposes the next one while making any
measurement less meaningful.

This is not specific to 26.06.01. Measured with `check_arch_support.sh`:

| Image | CUDA | TE | `sm_107a` in `ptxas` | TE cubins | bf16 | fp8/nvfp4 | `torch.compile` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nemo:26.06.01` | 13.2.1 | 2.16.0 | no | `…sm_103a sm_120` | OK | fail | fail |
| `nemo:26.08.00` | 13.3.0 | 2.17.1 | no | `…sm_103a sm_120` | OK | fail | fail |
| `nemo:dev` | 12.8.1 | — | no | — | — | — | — |
| `cuda:13.3.1-devel` | 13.3.1 | — | no | — | — | — | — |

`nemo:dev` is a stale tag (built 2025-04), older than the release tags — not a
nightly. And CUDA 13.3.1, the newest public toolkit, still has no `sm_107`
target: its list goes `sm_103 → sm_110`. The host driver here reports CUDA 13.4,
i.e. **the driver is ahead of every publicly released toolkit**, which is the
signature of pre-release silicon.

So the requirement is an early-access/internal container built on CUDA ≥ 13.4
with kernels for the arch. The launcher takes any image:

```bash
RUN_CONF_IMAGE=<image> ./launch_local.sh
```

Two concrete things to ask a vendor contact for, in preference order:

1. A NeMo container built on a toolkit that defines `sm_107`.
2. Failing that, a TransformerEngine built for the **family** target `sm_103f`
   rather than arch-exact `sm_103a`. Per `ptxas --help`, a `sm_XYf` binary runs
   on `sm_XZ` where `Z >= Y` in the same family, so `sm_103f` covers 10.7 —
   and CUDA 13.3 can already emit it. That alone would unblock fp8/nvfp4
   (Triton would still need `TORCHDYNAMO_DISABLE=1`).

Everything above is still worth keeping as a diagnostic sequence: it tells you
*which* layer is missing your arch, which is the fastest way to decide whether a
given container is usable at all.

## `grad norm` logging crash (upstream bug, any GPU)

Once a step actually runs, the first logged iteration can die in the *logger*:

```
File ".../megatron/bridge/training/utils/train_utils.py", line 1212, in training_log
    log_string += f" grad norm: {grad_norm:.3f} |"
TypeError: unsupported format string passed to Tensor.__format__
```

This is not arch-related. In Megatron-LM's `clip_grads.py::get_grad_norm_fp32`:

```python
if multi_tensor_scale_tensor_impl is not None:
    total_norm = total_norm.pow(1.0 / norm_type)          # stays a Tensor
else:
    total_norm = total_norm.item() ** (1.0 / norm_type)   # becomes a float
```

The tensor branch skips a device sync, but Megatron-Bridge's `training_log`
formats `grad_norm` with `:.3f`, which `Tensor.__format__` rejects. Both halves
ship inside the same container image, so this fires on any GPU that takes the
tensor branch. Worth reporting upstream — the fix is a `float(grad_norm)` in
`training_log`.

**Config-level avoidance does not work.** Both of these look like they should
help, and neither does — verified in the saved `ConfigContainer.yaml`, which
confirmed `optimizer.clip_grad = 0.0` and
`train.skip_sync_grad_norm_across_mp = True` were both applied, with the crash
unchanged:

* `optimizer.clip_grad=0.0` — the two `step()` implementations in
  `megatron/core/optimizer/optimizer.py` gate on `clip_grad > 0.0` and would
  yield `0.0` or `None`, but at least one other path produces the tensor anyway.
* `train.skip_sync_grad_norm_across_mp=true` — avoids
  `reduce_max_stat_across_model_parallel_group` re-wrapping the value, but the
  tensor does not originate there either.

So use `COMPAT_SHIM=true`, which runs the entrypoint through
[`compat_shim_local.py`](compat_shim_local.py). That patches
`torch.Tensor.__format__` to honour a float spec on a single-element tensor,
fixing every occurrence at once:

```bash
COMPAT_SHIM=true ... ./launch_local.sh
```

This is preferable to the config route on accuracy grounds too: it needs no
change to clipping or to grad-norm reduction, so the run stays numerically
faithful and the step time is not flattered by skipped work. The patch cannot
alter a computed value — it only changes how an already-computed scalar is
rendered into a log line, and it leaves multi-element tensors and empty format
specs alone.

## Container privileges

Beyond the flags NVIDIA recommends for the NeMo container (`--gpus`,
`--ipc=host`, `--ulimit memlock=-1`, `--ulimit stack=67108864`), the launcher adds:

* `--cap-add=SYS_NICE` — always. `numactl`'s `set_mempolicy`/`--membind` needs
  it, and Docker drops it by default. Without it every rank exits 1 with
  `set_mempolicy: Operation not permitted`.
* `--cap-add=SYS_ADMIN --cap-add=SYS_PTRACE` — only with `ENABLE_PROFILE=true`,
  so Nsight can sample. Depending on the host you may also need
  `sysctl -w kernel.perf_event_paranoid=1`.
* `--network=host` — single-node `torchrun` rendezvous on `127.0.0.1`.

The `--gpus` device list is passed as `"device=0,1,2,3"` with *literal* inner
quotes; without them Docker splits the comma list and rejects it with `cannot
set both Count and DeviceIDs on device request`.

`HF_TOKEN` is forwarded by name (`docker run -e HF_TOKEN`) rather than written
into `env.list`, so the token does not land in the results directory. It is also
deliberately not passed as `--hf_token` on the command line, where it would show
up in `ps`; the flag is launcher-only upstream and its only rank-visible effect
is `TRANSFORMERS_OFFLINE=0`, which `env.list` sets directly.

## Profiling

```bash
ENABLE_PROFILE=true JOB_TOTAL_GPUS=4 GPU_TYPE=vr200 MODEL_SIZE=8b ./launch_local.sh
```

Profiles steps 45–50 for all ranks by default (`PROFILE_START_STEP`,
`PROFILE_STOP_STEP`, `PROFILE_RANKS`), writing
`profile_<pid>_node0_rank<N>.nsys-rep` into `nsys_profile/` in the run
directory. `ENABLE_GPU_METRICS=true` adds device metrics.
`ENABLE_PYTORCH_PROFILE=true` selects the PyTorch profiler instead; the two are
mutually exclusive.

## Reading results later

```bash
python3 parse_results_local.py <run_dir>/log-<exp>.out
```

`--min-iteration`/`--max-iteration` widen or shift the averaging window, e.g. to
get a number out of a short `MAX_STEPS` run. Anything other than 35–44 is not
comparable to published results.
