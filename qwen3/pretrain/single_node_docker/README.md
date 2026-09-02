# Qwen3 pretrain: single-node Docker launch (no Slurm)

`../launch.sh` requires Slurm. These three scripts run the same workload on one
node with Docker and `torchrun`:

```bash
export LLMB_INSTALL=/mnt/nvme/llmb
export HF_TOKEN=<your token>          # Qwen3 repos are public, but HF_TOKEN is still read

./setup_local.sh                      # clone + image + caches
./launch_local.sh                     # normal launch
./run_bf16_fallback.sh                # launch on a GPU the container lacks kernels for
```

They are thin wrappers. The implementation is recipe-agnostic and lives once, in
[`../../../llama3.1/pretrain/megatron_bridge/single_node_docker/`](../../../llama3.1/pretrain/megatron_bridge/single_node_docker/),
selected via `RECIPE_DIR`. These wrappers point it at `qwen3/pretrain` and
default `MODEL_SIZE=30b`, so the tooling is discoverable from the recipe it
applies to. Anything the shared launcher accepts works here unchanged.

**Read first:**
[QUICKSTART.md](../../../llama3.1/pretrain/megatron_bridge/single_node_docker/QUICKSTART.md)
for host setup (proxy, image store, credentials) and
[README.md](../../../llama3.1/pretrain/megatron_bridge/single_node_docker/README.md)
for what is replaced on the Slurm side.

## What is specific to Qwen3

Two things are *easier* than the llama recipe:

* `Qwen/Qwen3-30B-A3B` is **not gated** — no access request, no `403`.
* 30b has a **native bf16 preset**, so no precision override is needed to avoid
  quantized kernels on an unsupported arch.

Three things are harder, all from it being mixture-of-experts.

### Expert parallelism must fit the node

The 30b preset is 8 GPUs with `EP=8`. Expert parallelism cannot exceed the ranks
available, so on four GPUs set `EP=4`. Global batch size still follows the
recipe's weak scaling (`gbs_scaling_factor × num_gpus` = 256 on four GPUs).

### The dispatcher assumes a multi-node fabric

The preset selects **HybridEP**, which is designed for an NVL72 domain; left
alone, the resolved environment exports `USE_MNNVL=1` and
`NVLINK_DOMAIN_SIZE=72` on a single node. Use `MOE_BACKEND=alltoall`, which
switches the dispatcher *and* keeps the emitted environment consistent with it.

Two details that make this non-obvious:

* `apply_flex_dispatcher_backend()`'s guard is
  `device_properties.major in [8, 9, 10]`. On a major-10 part HybridEP therefore
  **engages** rather than being skipped for lack of support — it has to be
  turned off deliberately.
* The override also sets `moe_flex_dispatcher_backend=null`, because that
  function leaves the field on the model config and other code branches on it.

### MoE routing and permutation are Triton

`moe_permute_fusion` and `moe_router_fusion` are additional **direct-Triton**
call sites. On a GPU whose arch the container's `ptxas` does not know, they
abort with `LLVM ERROR: Cannot select ... shfl.sync.bfly.i32`, exactly like
TransformerEngine's fused cross-entropy. These are extra *call sites*, not a
different root cause, and `TORCHDYNAMO_DISABLE=1` does **not** cover them
because they are not `torch.compile`.

### Do not copy the dense model's recompute settings

`recompute_granularity=full` fails here with

```
AssertionError: full recompute is only supported with full iteration CUDA graph
```

because the qwen3 preset uses `cuda_graph_impl=transformer_engine` scoped to
attn/moe_router/moe_preprocess. It is also unnecessary: expert parallelism
shards the experts and nothing runs out of memory at `MBS=1`.

## Measured on 4 × compute-capability 10.7

`./run_bf16_fallback.sh`, 50/50 iterations, loss falling 12.34 → 8.13:

```
 Averaging window:   iterations 35-44
 Samples:            10
 s/iter:             28.135 (std 0.340)
 TFLOPS/GPU:         214.38 (std 2.58)
```

> **Not a benchmark result.** bf16 rather than fp8, eager instead of fused
> activations, unfused attention, MoE routing/permutation fusions disabled,
> `MBS=1`, and 4 GPUs against a validated minimum of 8 — every one a slowdown.
> It shows the stack runs on this silicon; it is not comparable to published
> Qwen3 figures.

Run `../../../llama3.1/pretrain/megatron_bridge/single_node_docker/check_arch_support.sh <image>`
first — if it reports `bf16 OK` with everything else failing, you need
`run_bf16_fallback.sh` rather than `launch_local.sh`.
