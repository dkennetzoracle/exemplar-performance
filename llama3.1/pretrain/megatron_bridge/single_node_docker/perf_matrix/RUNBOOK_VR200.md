# Runbook: reproduce this on a Vera Rubin (sm_107) node

**Note to self.** The GB300 (sm_103) side of this comparison is done — see
[`RESULTS.md`](RESULTS.md). This is the other half: run the same matrix on a
Vera Rubin node to (a) re-confirm the VR200 reference on current tooling and
(b) find out whether the arch-independent levers that helped on GB300 also help
a machine that is stuck on the bf16 fallback.

The GB300 cluster used for the first half was 1,482 nodes of `gpu:B300:4` with
no VR200 in it, so nothing below had been verified on sm_107 when it was
written — everything was a hypothesis with a measured GB300 counterpart.

> **Execution status on a 4 × sm_107 node (2026-09-03).** Section 0 and step 1
> are **done and reproduced**; results are folded into `RESULTS.md`. Steps 2
> and 3 are in progress. Per-step outcomes are recorded inline below.

---

## What is already established about sm_107

From [`../arch_support/nemo-26.06.01.md`](../arch_support/nemo-26.06.01.md) and
[`nemo-26.08.00.md`](../arch_support/nemo-26.08.00.md), re-verified against a
GB300 probe of the *same two images*:

| | VR200 (sm_107) | GB300 (sm_103) |
| --- | --- | --- |
| `ptxas` accepts `sm_<cc>a` | **NO** | YES |
| `ptxas` target list | `sm_100 sm_100a sm_103a sm_110 sm_110a sm_120 sm_121 …` | **identical list** |
| TE cubin archs | `sm_100 sm_100a sm_103a sm_120 sm_75 sm_80 sm_89 sm_90 sm_90a` | **identical list** |
| `te_ptx_entries` | 0 | 0 |
| bf16 / fp8_cs / fp8_mx / nvfp4 | OK / FAIL / FAIL / FAIL | OK / OK / OK / OK |
| `torch.compile` | FAIL | OK |

The two lists are byte-identical between the machines and between the two
container versions. `sm_103a` is in both; `sm_107a` is in neither. Note the
list already contains `sm_110`/`sm_121`, so the `sm_107` omission looks like a
gap in the target list rather than a toolchain that predates the part.

**Confirm this still holds before running anything** — if a newer container
fixed it, the whole plan changes:

```bash
./check_arch_support.sh nvcr.io/nvidia/nemo:26.06.01
./check_arch_support.sh nvcr.io/nvidia/nemo:26.08.00
```

**Result (2026-09-03, 4 × sm_107, driver 615.62): unchanged on both images.**
`ptxas` still has no `sm_107a`; TE cubins still
`sm_100 sm_100a sm_103a sm_120 sm_75 sm_80 sm_89 sm_90 sm_90a` with
`te_ptx_entries=0`; bf16 OK, fp8_cs / fp8_mx / nvfp4 / `torch.compile` all
FAIL. Nothing a newer container fixed, so the plan below stands as written.

---

## Run order

```bash
cd llama3.1/pretrain/megatron_bridge/single_node_docker/perf_matrix
./node_prep.sh                                    # read README: containerd, not data-root
IMAGES="26.06.01 26.08.00" ./setup_images.sh
```

### 1. Reference — re-confirm the published VR200 numbers

```bash
IMAGE_TAG=26.06.01 ./run_matrix.sh reference
```

Expect to land on the recorded values. If these do not reproduce, stop and
find out why before trusting anything else:

| Workload | s/iter | TFLOPS/GPU | tokens/s/GPU |
| --- | --- | --- | --- |
| llama3.1 8B | 2.504 | 336.82 | 6,543 |
| Qwen3 30B-A3B | 28.135 | 214.38 | 9,317 |

Qwen3's loss should fall 12.34 → 8.13 over the 50 steps; on GB300 it matched to
four significant figures, so it is a good same-config check.

**Result: both reproduced.**

| Workload | recorded tok/s/GPU | re-measured | delta |
| --- | --- | --- | --- |
| llama3.1 8B | 6,543 | **6,543** (2.504 s/iter, 336.81 TFLOPS) | exact |
| Qwen3 30B-A3B | 9,317 | **9,300** (28.189 s/iter, 213.98 TFLOPS) | −0.2% |

Qwen3 loss 12.34119 → 8.136611, matching the recorded 12.34 → 8.13. The 0.2%
on Qwen3 is run-to-run noise (`s/iter` std 0.148 on that row, i.e. ±0.5%), not
a regression: these rows carry `train.eval_iters=0`, which changes only the
post-training eval, not the timed iterations.

### 2. Native — capture the failure as evidence, once

```bash
IMAGE_TAG=26.06.01 ./run_matrix.sh native
```

Every quantized row is **expected to fail** with `fp8_dpa_no_backend` or
`no_cubin_for_arch`. That is the point: it produces per-image, per-config
evidence for the ticket. Do not spend time working around it. Repeat on
`IMAGE_TAG=26.08.00` to show the newer container does not fix it.

### 3. Portable — the actual open question

```bash
IMAGE_TAG=26.06.01 ./run_matrix.sh portable
IMAGE_TAG=26.08.00 ./run_matrix.sh portable
./collect_results.py
```

Five levers, all bf16, none needing a kernel the container might lack.
**Ordered by measured or expected size — run them top-down if time is short.**

| Lever | GB300 measurement | Hypothesis for VR200 |
| --- | --- | --- |
| **Qwen3 MBS=1 -> 2 (and 4)** | **1.91x measured** (7,860 -> 15,042 tok/s/GPU, bf16, same container) | **Run this first.** By far the largest portable win found, and it is pure batch shape — no kernel involved, so it should transfer directly. The VR200 reference is MBS=1, so this lever is untouched there. MBS=4 is untested on either machine (the GB300 attempt died with the node); MBS=8 OOMs, so 4 is the last candidate. |
| **Drop full recompute (llama)** | untested on either | Full recompute was added *because* unfused attention materialises the score matrix, and it costs ~30% extra compute. At MBS=1/seq 8192 that tensor is ~4 GiB on a ~280 GB GPU. If it is not needed, this is the largest remaining llama win. |
| **llama MBS=2 / MBS=4** | MBS=2: 3.457 -> 3.410 s/iter (+1.4%) | QUICKSTART says MBS=1 is *required* because unfused attention OOMs at MBS=2. That **did not reproduce** on a 284 GB GB300 with the identical unfused-attention config. MBS=4 in the *fallback* path is untested — worth trying given MBS dominated everywhere else. |
| **Larger GBS (32, 64)** | GA=1 -> 4 gave +4.0% on nvfp4 | Amortises the optimizer step and DP all-reduce over more microbatches. No arch dependency. Applies to llama (reference is GBS=8); Qwen3 is already GBS=256/GA=64, so GBS has little headroom there — for Qwen3 the lever is MBS, not GBS. |
| **`nemo:26.08.00`** | bf16-fallback 3.457 -> 3.400 s/iter (+1.7%) | Free if it works, and bf16 is confirmed OK on sm_107 for that image. Compare with **`s/iter`, not TFLOPS** — see the accounting caveat in the README. |

The Qwen3 MBS lever is the one that changed since this runbook was first
written. The GB300 MoE sweep found that **MBS dominates precision** for
Qwen3 30B-A3B at `EP=4`: raising MBS nearly doubled throughput, while
switching bf16 -> fp8 at MBS=1 made it **37% slower**. VR200 cannot run fp8
at all, but it can raise MBS, so this is a portable win that costs nothing
but a config change. If it reproduces, VR200's own Qwen3 ceiling moves well
above the 9,317 tok/s/GPU reference.

---

## What is *not* portable, and why that is the headline

Do not spend effort trying to recover these on sm_107 — each is blocked by a
missing kernel, not by configuration:

| Improvement | GB300 gain | Blocker on sm_107 |
| --- | --- | --- |
| nvfp4 / fp8 instead of bf16 | bulk of **9.66×** | no TE cubin, `te_ptx_entries=0` → no JIT fallback |
| cuDNN fused attention | part of 9.66× | no execution plan built for the arch |
| `torch.compile` / jit_fuser, fused cross-entropy | part of 9.66× | `ptxas` has no `sm_107a` target |
| MoE permute/router fusion | ~11% on Qwen3 | Triton → same `ptxas` block |

The 9.66× is measured, not extrapolated: on GB300, at **identical batch shape**
(MBS=2, GBS=8, same node, same container), the bf16-fallback path ran at
3.410 s/iter and the native nvfp4 path at 0.353 s/iter. That single ratio is
the cost of the missing kernels.

Which produces the sharp framing for NVIDIA: in the one path VR200 *can*
execute, it beats GB300 by ~32–38% — Rubin's bf16 silicon is genuinely faster.
It is then blocked out of the path that is ~9.7× better on otherwise-slower
hardware.

---

## Filing: four independent items

Keep these separate; different teams own them.

1. **TransformerEngine ships no `sm_107a` cubins and no PTX.** Every quantized
   precision hard-fails. Ask: build `sm_107a` cubins, and/or embed PTX so an
   unlisted arch degrades to a JIT compile instead of a hard failure. Evidence:
   `arch_support/*.md`, `te_cubin_archs` + `te_ptx_entries=0`.
2. **Bundled CUDA `ptxas` has no `sm_107a` target.** Independent of TE; breaks
   Triton and `torch.compile`. Unchanged across CUDA 13.2.1 (26.06.01) and
   13.3.0 (26.08.00) while `sm_110`/`sm_121` are present.
3. **No TE fp8-attention backend, affecting sm_103 too.** Every nvfp4/fp8
   preset sets `fp8_dot_product_attention=true` and dies with `No dot product
   attention backend is available` — reproduced on GB300 at MBS=1 and MBS=2.
   One-line workaround, but the shipped presets are broken for 4-GPU GB300.
4. **`nemo:26.08.00` CUDA-graph capture assert in the eval path.** PyTorch's own
   `INTERNAL ASSERT` text asks for a bug report. Training unaffected;
   `train.eval_iters=0` avoids it. Not present in 26.06.01 (torch 2.12.0a0).

Internal-only, not an NVIDIA ask: Megatron-Bridge's model-FLOPs accounting
changed ~4.6% between the two containers, so any historical TFLOPS trend
spanning them has a discontinuity that is not hardware.

---

## Reporting back

`./collect_results.py` writes `perf_matrix.csv` into `$RESULTS_DIR` and prints a
table ranked by `tokens/s/GPU` with a VR200-relative column. Copy the CSV off
the node — `$RESULTS_DIR` defaults to node-local disk.

Worth capturing beyond the CSV:

- `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` on one failing quantized run, for TE's own
  account of why it declined every attention backend. Attach to item 3.
- Whether dropping recompute changed the loss curve, not just the speed.
- `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version`, since
  the driver version matters for the bf16 path (cuBLAS/cuDNN track the driver,
  which is why bf16 works at all on an arch the container has no cubins for).
