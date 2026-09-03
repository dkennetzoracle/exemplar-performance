# Issues surfaced by the cross-arch perf matrix

Everything below was hit while running the matrix on 4 × GB300 (sm_103) and
4 × Vera Rubin (sm_107), `nemo:26.06.01` and `nemo:26.08.00`, Megatron-Bridge
`b50da4c`. Each item says who owns it and whether it needs a new container,
because those two questions get conflated and the answer is mostly "no".

**Summary:** of the twelve items, **three** require a container rebuild. The
rest are upstream code or preset bugs, our own tooling, or our own analysis
errors — all fixable without waiting on anyone.

Status legend: **OPEN** needs someone else · **WORKAROUND** running with a
documented override · **FIXED** fixed in this repo.

---

## A. Architecture coverage — needs a rebuilt container

These are the only items that block on NVIDIA. Together they are the entire
gap between VR200's tuned bf16 ceiling (9,635 tok/s/GPU) and GB300's tuned
nvfp4 result — a **6.16×** dense-model difference on otherwise slower silicon.

### A1. TransformerEngine ships no `sm_107` cubin and no PTX — **OPEN**

Every quantized precision hard-fails in its quantize kernel before any GEMM:

```
fp8_cs  fp8/quantize_fp8.cuh:385      quantize_1D
fp8_mx  mxfp8/quantize_mxfp8.cuh:828  quantize
nvfp4   hadamard_transform.cu:599     hadamard_transform_amax
        -> CUDA Error: no kernel image is available for execution on the device
```

`cuobjdump --list-elf` gives `sm_75 sm_80 sm_89 sm_90 sm_90a sm_100 sm_100a
sm_103a sm_120` — identical in both containers — and `--list-ptx` gives **0
entries**, so there is no JIT fallback and the arch list is exhaustive.

**The cheap fix, and the only one measured to work: build TE for the family
target `sm_103f` rather than arch-exact `sm_103a`.** Compiling a kernel that
uses `__shfl_xor_sync` — the exact intrinsic Triton also fails on — and
checking the computed value, not merely the absence of a crash:

| build target | result on sm_107 |
| --- | --- |
| `sm_103a` (what TE ships) | `no kernel image is available` — reproduces A1 verbatim |
| **`sm_103f`** (family) | **correct result** |
| `sm_103` (plain) | correct result |
| `sm_110f` | `no kernel image is available` — different family |
| `compute_103` PTX, driver-JIT | correct result |

CUDA 13.3 already emits `sm_103f`, so **this needs no new toolchain** — only a
build-flag change (`NVTE_CUDA_ARCHS`). Asks, in order: (a) `sm_103f` family
build; (b) embed PTX so an unlisted arch degrades to a JIT compile instead of a
hard failure — a structural fix rather than a per-arch chase; (c) explicit
`sm_107a` cubins.

Evidence: [`../arch_support/nemo-26.06.01.md`](../arch_support/nemo-26.06.01.md),
[`nemo-26.08.00.md`](../arch_support/nemo-26.08.00.md).

### A2. Bundled CUDA `ptxas` has no `sm_107` target — **OPEN**

Independent of A1. Breaks Triton and therefore `torch.compile`, Megatron's
`jit_fuser`, TE's fused cross-entropy, and MoE router/permute fusion:

```
'sm_107a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32
```

Unchanged across CUDA 13.2.1 (26.06.01), 13.3.0 (26.08.00) and 13.3.1, while
`sm_110` and `sm_121` are both present — so this reads as a gap in the target
list, not a toolchain that predates the part.

**Two things must change together.** `ptxas` needs the target, *and* Triton's
bundled LLVM must recognise it: LLVM currently writes `.target sm_107a` into
the PTX while itself warning `sm_107a is not a recognized processor`. That
string lives in `triton/_C/libtriton.so` (2 occurrences) and **not** in
`ptxas` (0), so fixing only `ptxas` would not be enough.

Patching Triton is not a local workaround either. `TRITON_OVERRIDE_ARCH`
accepts only `^sm(\d+)$`, so it cannot express a family target, and
monkeypatching `sm_arch_from_capability` redirects `ptxas --gpu-name` but
leaves the PTX `.target` at `sm_107f`, which `ptxas` then rejects. Triton's own
source carries `# TODO: Handle non-"a" sms`, so upstream family-target support
would be the alternative route.

### A3. cuDNN builds no fused-attention plan for the arch — **OPEN**

```
fused_attn_f16_arbitrary_seqlen.cu:422
  cuDNN Error: [cudnn_frontend] Error: No valid execution plans built.
```

**This is the expensive one, and its cost is indirect.** Forcing
`NVTE_UNFUSED_ATTN=1` makes attention materialise the score matrix, which
scales with MBS — so it is what caps MBS at 1 and made full recompute look
necessary. It is also why the GB300 MBS finding does not transfer (see D2).
Fixing A3 would unlock the MBS lever on sm_107, which was worth up to 3.10× on
GB300.

---

## B. Upstream code and preset bugs — no container needed

### B1. The nvfp4 presets are internally inconsistent, on *every* arch — **WORKAROUND**

Originally recorded as "TE has no fp8 attention backend for sm_103", implying
an arch gap. It is not. TE
`pytorch/attention/dot_product_attention/utils.py:607`:

```python
if use_fused_attention and (fp8_recipe.float8_block_scaling() or fp8_recipe.nvfp4()):
    logger.debug("Disabling FusedAttention for %s", fp8_recipe.__class__.__name__)
    use_fused_attention = False
```

No compute-capability condition — while every neighbouring branch has one
(`arch >= sm100` at 572, `arch < sm100` at 578 and 595, `== (12, 0)` at 611,
`sm120` at 615). So TE demonstrably arch-gates elsewhere and this disable is
unconditional on the *recipe*. Confirmed independently in the sm_107 container
after the sm_103 finding, i.e. from two architectures.

Every nvfp4 preset sets `fp8_dot_product_attention=true`, so all three backends
self-disable and the run dies with `No dot product attention backend is
available`. **The shipped preset cannot start on any architecture — GB200 and
B200 included.** Owner: Megatron-Bridge presets (or TE, if the disable is
wrong). Workaround: `mixed_precision.fp8_dot_product_attention=false
model.fp8_dot_product_attention=false`.

Evidence: [`evidence/fp8_dpa_nvfp4_no_backend.md`](evidence/fp8_dpa_nvfp4_no_backend.md).

### B2. `grad_norm` tensor formatted with a float spec — **WORKAROUND**

```
train_utils.py:1212  log_string += f" grad norm: {grad_norm:.3f} |"
TypeError: unsupported format string passed to Tensor.__format__
```

Megatron-LM's `get_grad_norm_fp32` returns a 0-dim tensor when
`multi_tensor_scale_tensor_impl` is set (it skips a `.item()` to avoid a device
sync); Megatron-Bridge's `training_log` formats it as a float. Both halves ship
in the same image, so this fires on **any** GPU, and it stops training at the
first logged iteration.

Config-level avoidance does not work: `optimizer.clip_grad=0.0` plus
`train.skip_sync_grad_norm_across_mp=true` were both verified applied (in the
saved `ConfigContainer.yaml`) with the crash unchanged. Upstream fix is a
`float()` in `training_log`. Workaround: `COMPAT_SHIM=true`, which patches only
the formatting and cannot change a computed value.

### B3. `nemo:26.08.00` CUDA-graph assert in the eval path — **WORKAROUND**

PyTorch `INTERNAL ASSERT FAILED` during CUDA-graph capture in **post-training
eval** (torch 2.13.0a0); absent in 26.06.01 (torch 2.12.0a0). Training itself
is unaffected, so timings from a run that reached eval are valid. PyTorch's own
message asks for a bug report. Workaround: `train.eval_iters=0`.

---

## C. Tooling bugs in this repo

### C1. Fallback detection keyed on a tag substring — **FIXED**

`collect_results.py` used `"fallback" in tag`. The GB300 rows were measured
with ad-hoc tags (`A1-llama31-8b-bf16fallback`) containing the word, but
`run_matrix.sh` — the driver the runbook tells you to use — emits
`...-bf16-...`, which does not. So every llama fallback row measured with the
actual driver fell through to the `quantized` branch (because `precision`
reads `nvfp4` on a bf16 run, since llama31_8b has no bf16 preset) and was
labelled *"VR200 cannot run it"* — for the exact config VR200 runs as its
reference.

Now detected by mechanism: `launch_local.sh` records the resolved container env
to `env.list` in the run dir and the log banner carries that path, so a
fallback row is the one with `NVTE_UNFUSED_ATTN=1`. Falls back to the tag
string when the run dir is unreachable, so GB300 rows still classify.

### C2. `collect_results.py` ignored `LLMB_INSTALL` — **FIXED**

Defaulted to a hardcoded `/mnt/localdisk/llmb/perf-matrix-results`, so the bare
`./collect_results.py` the runbook instructs you to run failed on every node
that is not the GB300 cluster. Now follows `lib.sh`'s precedence: `RESULTS_DIR`,
else `$LLMB_INSTALL/perf-matrix-results`.

### C3. `node_prep.sh` can destroy a working Docker proxy config — **OPEN**

`IMAGE_STORE` defaults to `/mnt/localdisk`. On any node whose containerd root
is elsewhere, that mismatch takes the relocation branch, which writes
`/etc/docker/daemon.json` from a heredoc containing only `default-ulimits` —
**dropping any `proxies` block**, which breaks `docker pull` on a proxied
network — and rsyncs the image store to a path that may not exist. It should
merge into the existing `daemon.json` rather than overwrite it, and refuse to
relocate unless `IMAGE_STORE` was set explicitly. Currently mitigated only by
passing `IMAGE_STORE` on every invocation.

### C4. Matrix gap: the two winning levers were never combined — **FIXED**

Recompute was varied only at GBS=8, and GBS only with recompute *on*, so the
setting that actually wins on sm_107 (MBS=1, no recompute) had never been
combined with the GBS lever. Added; the gains stack, and that is where the
1.44× ceiling comes from.

---

## D. Analysis errors, corrected by measurement

Kept because both were confidently asserted in docs before being measured.

### D1. Full recompute was never needed — **FIXED**

`MBS=1` and full recompute were set together while debugging an OOM and never
separated. `MBS=1` alone was sufficient; the recompute cost **40.7%** and had
been baked into `run_bf16_fallback.sh` and QUICKSTART as required.

The `MBS=1` claim needs restating rather than deleting: MBS=2 and MBS=4 *do*
run with recompute (+1.8% / +3.4%) and OOM without it. Accurate form: **"MBS>1
requires recompute, and recompute costs more than the higher MBS returns."**

### D2. The MBS reprioritisation did not transfer to sm_107 — **FIXED**

Qwen3 MBS=1→4 measured 3.10× on GB300 and was promoted to the top VR200 lever
as *"pure batch shape, no kernel involved, so it should transfer directly"*.
Both VR200 MBS rows OOM.

The 3.10× is a `nat-` row, and `nat-` rows carry no `FALLBACK_ENV`, so they run
cuDNN **fused** attention; `por-` rows force `NVTE_UNFUSED_ATTN=1`, and unfused
attention materialises the score matrix, which scales with MBS. In the fallback
path MBS is entangled with the one kernel sm_107 lacks (A3). **A GB300 `nat-`
measurement cannot predict a VR200 `por-` result** — only `por-` rows are
portable evidence, which is what the row sets exist for.

---

## What this means for planning

- **bf16 works today and is tunable.** VR200's ceiling moved from 6,543 to
  9,405 tok/s/GPU (1.44×) with no container change, purely by dropping
  recompute and raising GBS. Nothing here is blocked on waiting.
- **The quantized path is blocked on A1–A3, and only those.** Everything in B
  is a preset or logic bug with a one-line workaround.
- **A1 is the single highest-leverage ask** because it is measured, needs no
  new toolchain, and is a build flag: `NVTE_CUDA_ARCHS` targeting `sm_103f`.
- **A3 is the sleeper.** It looks like "one more missing kernel", but it is
  what forces unfused attention, which caps MBS at 1, which is why the largest
  lever found on GB300 is unavailable here.
- **B1 should be filed against the presets, not the container**, and it affects
  GB200/B200 too — the ticket is "the nvfp4 preset cannot start on any
  architecture", not "add sm_103 kernels".
