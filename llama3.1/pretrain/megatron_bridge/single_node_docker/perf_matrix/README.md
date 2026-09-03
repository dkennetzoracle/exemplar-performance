# Perf matrix: cross-architecture single-node comparison

Reproducible driver for comparing one recipe across GPU architectures on a
single node with Docker, and for finding the best configuration a given
node can actually reach.

Built to answer three separate questions, which need three different row sets
because they are not comparable to each other:

| Mode | Question | Runs on |
| --- | --- | --- |
| `reference` | What does the *same* config do on both machines? | any arch (bf16 only) |
| `native` | What is the best this machine can do? | archs with TE cubins |
| `portable` | Can a bf16-only machine be tuned without new kernels? | any arch |

`reference` is the apples-to-apples row. On an architecture the container ships
no kernels for, it is also the *ceiling* — not a handicap — so it is the honest
baseline for that machine.

---

## Quick start

All three scripts run **on the GPU node**, not on a login/controller node.

```bash
cd llama3.1/pretrain/megatron_bridge/single_node_docker/perf_matrix

./node_prep.sh                                   # once per node
IMAGES="26.06.01 26.08.00" ./setup_images.sh     # once per node
./run_matrix.sh reference
./collect_results.py
```

Under Slurm, wrap each command rather than running on the controller:

```bash
salloc --no-shell -p compute -N1 --gres=gpu:4 --exclusive -t 480
srun --jobid=<id> --overlap -N1 bash ./run_matrix.sh reference
```

Credentials come from files so keys never reach shell history or `env.list`:
`.hf` (HuggingFace) and `.nv` (NGC) at the repo root. Both are gitignored.

Rows are skipped when `$RESULTS_DIR/<tag>.done` exists, so every script is
resumable — fix one failure and re-run without repeating the rest.

---

## Reading the numbers: pick the right metric

Two of the three obvious metrics are traps.

**`TFLOPS/GPU` is not comparable across container versions.** The value is
parsed from the framework's own `MODEL_TFLOP/s/GPU` log line, not computed.
Megatron-Bridge's model-FLOPs accounting changed between `nemo:26.06.01` and
`nemo:26.08.00`: two unrelated configs both showed the newer container
reporting a factor **0.953** of the older one's FLOPs per iteration. On the
bf16-fallback row this actively inverts the conclusion — 26.08.00 was **1.7%
faster by the stopwatch** while reporting **3.0% lower TFLOPS**.

**`s/iter` is honest wall clock but scales with global batch size**, so it
cannot compare rows with different `GBS`.

**`tokens/s/GPU` = `GBS × seq_len / s_iter / gpus` is immune to both**, and is
what `collect_results.py` ranks by. Where all three metrics are valid it
reproduces the TFLOPS-derived delta exactly, which is the check that it is not
introducing its own distortion.

Use `TFLOPS/GPU` only *within* one container — for MFU and for precision/batch
sweeps. Use `tokens/s/GPU` for anything crossing a container or a batch shape.

---

## Known failure modes

`collect_results.py` keeps failed configs as rows and classifies each, so an
empty metric cell always comes with a reason. Codes it emits:

| Code | Meaning and fix |
| --- | --- |
| `no_cubin_for_arch` | TE ships no cubin for the arch and `te_ptx_entries=0`, so there is no PTX to JIT from and the launch fails outright. bf16 only — use `reference`/`portable`. |
| `ptxas_no_target` | Bundled `ptxas` has no `sm_<cc>a` target, so Triton and `torch.compile` cannot build. `TORCHDYNAMO_DISABLE=1`. |
| `cudnn_no_attn_plan` | cuDNN built no fused-attention plan for the arch. `NVTE_UNFUSED_ATTN=1`. |
| `fp8_dpa_no_backend` | TE has no **fp8 attention** backend for the arch, and every nvfp4/fp8 preset sets `fp8_dot_product_attention=true`. Override both `mixed_precision.` and `model.` copies to `false`. Affects sm_103 as well as sm_107. |
| `triton_cross_entropy` | TE's fused cross-entropy is Triton and aborts uncatchably. `model.cross_entropy_loss_fusion=false`. |
| `oom_dense` | Activation memory at this MBS exceeds device memory. MBS=4 was the ceiling for llama3.1 8B on 4 × 284 GB. |
| `oom_moe` | The `gb300` preset assumes `EP=8` across 8 GPUs, so `EP=4` on one node doubles expert weights per GPU. Lower MBS. |
| `preset_is_8gpu_shape` | Preset has `TP*PP*CP=8` and cannot shard over 4 GPUs. Pin `TP=1 PP=1 CP=1` — a deviation to report alongside the number. |
| `no_such_preset` | No preset for that `gpu_type`/`dtype`/`variant`. `llama31_8b` has **no bf16 preset on any GPU type**, which is why `reference` selects the nvfp4 preset and overrides the precision. |
| `cuda_graph_assert` | `nemo:26.08.00` (torch 2.13.0a0) hits a PyTorch `INTERNAL ASSERT` in CUDA-graph capture during the **post-training eval**. Training is unaffected, so timings from such a run are valid — the row is marked `OK (eval crashed)`. `train.eval_iters=0` avoids it. |

---

## Host prep, and the one trap that costs an image pull

`node_prep.sh` handles both of these; they are documented here because they are
easy to get wrong by hand.

**Docker 25+ keeps images in *containerd's* root, not Docker's `data-root`.**
Setting `data-root` in `daemon.json` looks like it worked — `docker info`
reports the new path — but images never go there. Measured on a Docker 29.6
node: `/mnt/localdisk/docker` held **4.2 MB** while `/var/lib/containerd` held
**70 GB**, and the second image pull died with `no space left on device`
writing to `/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/`. The
knob that actually moves images is `root` in `/etc/containerd/config.toml`.
Each NeMo image is ~60–65 GB unpacked, so two of them will not fit on a 123 GB
root disk.

**Core dumps fill the root filesystem.** A rank with ~280 GB of device memory
mapped writes a core of roughly that size; one failed 4-rank run put 4 × 17 GB
into `/var/lib/apport/coredump`. Bring-up means crashes, so `node_prep.sh` sets
`default-ulimits.core = 0` for containers.

**The repo parser needs `typer`.** `parse_results_local.py` imports
`llmb_run.pretrain_log_parser` from `cli/llmb-run`. Without its deps every run
trains fine and then prints `could not import llmb_run.pretrain_log_parser`.
`setup_images.sh` builds a venv for it; `lib.sh` puts it on `PATH`. If runs
already finished without metrics, `./backfill_metrics.sh` re-parses them
in place (must run on the node that holds the run directories).

---

## Files

| File | Purpose |
| --- | --- |
| `node_prep.sh` | containerd root relocation, core-dump guard, arch report |
| `setup_images.sh` | nvcr.io login, HF access check, `setup_local.sh` per recipe/image, parser venv |
| `run_matrix.sh` | the three row sets (`reference` / `native` / `portable` / `all`) |
| `lib.sh` | shared `run()` helper, the workaround definitions, path resolution |
| `collect_results.py` | logs → CSV + ranked table with cause attribution |
| `backfill_metrics.sh` | re-parse finished runs missing metrics |
| `RUNBOOK_VR200.md` | note-to-self for reproducing this on a Vera Rubin (sm_107) node |
| `RESULTS.md` | measured numbers, with the config each came from |

The workaround strings in `lib.sh` are deliberately spelled out rather than
delegated to `run_bf16_fallback.sh`, so that individual workarounds can be
varied per row — which is what `portable` mode is for. Keep them in sync with
[`../run_bf16_fallback.sh`](../run_bf16_fallback.sh), whose `WHY` block holds
the rationale for each.
