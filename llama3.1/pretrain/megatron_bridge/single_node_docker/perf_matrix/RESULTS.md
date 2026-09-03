# Measured results: GB300 (sm_103) vs Vera Rubin (sm_107)

All rows: **4 GPUs, one node, 50 steps, mock data**, Megatron-Bridge
`b50da4c7404caa41793e74ac40d18798844c7b67`.

* **VR200** = compute capability 10.7, aarch64. Reference values from the
  "Verified state" sections of [`../README.md`](../README.md) and qwen3's
  `single_node_docker/README.md`.
* **GB300** = `NVIDIA GB300`, compute capability 10.3, 284 GB HBM,
  driver 595.71.05, 144 × Grace (2 sockets), OCI Slurm cluster.

> Not benchmark configurations. 4 GPUs is below the recipe's validated minimum
> of 8 for both workloads, and several rows carry documented deviations. Do not
> compare these to published figures.

## Sign convention

`VR200 advantage` is positive when VR200 is faster, computed ratio-wise as
`VR200 / GB300 - 1` on `tokens/s/GPU`. Where GB300 is faster the column shows a
multiplier instead. Where a machine cannot execute a config at all the cell is
`N/A — no kernel`, not a percentage: there is no baseline to divide by.

`tokens/s/GPU` is the primary metric — `TFLOPS/GPU` is not comparable across
container versions (see the accounting caveat in [`README.md`](README.md)).

---

## 1. Same config, both machines — the apples-to-apples rows

bf16 fallback, `nemo:26.06.01`, all six workarounds. **On VR200 this is the
ceiling, not a handicap**: TE ships no `sm_107a` cubin and no PTX, so nothing
quantized runs at any setting.

| Workload | VR200 | GB300 | VR200 advantage |
| --- | --- | --- | --- |
| llama3.1 8B | 2.504 s/iter · 336.82 TFLOPS · **6,543** tok/s/GPU | 3.457 s/iter · 243.91 TFLOPS · **4,739** tok/s/GPU | **+38.1%** |
| Qwen3 30B-A3B | 28.135 s/iter · 214.38 TFLOPS · **9,317** tok/s/GPU | 37.202 s/iter · 162.12 TFLOPS · **7,047** tok/s/GPU | **+32.2%** |

**VR200 wins the one path it can execute, on both a dense and an MoE model.**
The bf16 fallback runs in cuBLAS/cuDNN, which ship with the *driver* rather than
the container, so it works on an arch the container has no cubins for — and in
that path Rubin's silicon is genuinely faster than Blackwell Ultra.

Both measurements are stable (std 0.001 and 0.143 s/iter) and Qwen3's loss curve
on GB300 matched the VR200 reference to four significant figures
(12.34591 → 8.138762 vs the recorded 12.34 → 8.13), confirming identical config.

### VR200 re-measured on current tooling (2026-09-03)

The VR200 column above was carried over from the original bring-up. Re-running
the same two rows on a 4 × sm_107 node with `run_matrix.sh reference`
reproduces it, so the cross-machine comparison rests on a repeatable baseline
rather than a single historical run:

| Workload | original | re-measured | delta |
| --- | --- | --- | --- |
| llama3.1 8B | 2.504 s/iter · 336.82 TFLOPS · 6,543 tok/s/GPU | 2.504 s/iter · 336.81 TFLOPS · **6,543** | exact |
| Qwen3 30B-A3B | 28.135 s/iter · 214.38 TFLOPS · 9,317 tok/s/GPU | 28.189 s/iter · 213.98 TFLOPS · **9,300** | −0.2% |

Qwen3 loss 12.34119 → 8.136611. The 0.2% is inside that row's own run-to-run
spread (`s/iter` std 0.148) — these rows set `train.eval_iters=0`, which touches
only post-training eval, not the timed iterations.

---

## 2. Best achievable per machine

| Workload | VR200 best | GB300 best | GB300 advantage |
| --- | --- | --- | --- |
| llama3.1 8B | 6,543 tok/s/GPU (bf16 — its ceiling) | **59,282** tok/s/GPU (nvfp4, 26.08.00, MBS=4 GBS=64) | **9.06x** |
| Qwen3 30B-A3B | 9,317 tok/s/GPU (bf16 — its ceiling) | **15,042** tok/s/GPU (bf16, 26.06.01, MBS=2 GBS=256) | **1.61x** |

The two workloads behave completely differently, and the dense-model headline
does **not** generalise. Anyone extrapolating 9x to a mixture-of-experts
workload would be wrong by nearly an order of magnitude.

### llama3.1 8B, full sweep (seq_len 8192)

| Container | Precision | MBS | GBS | s/iter | TFLOPS/GPU | tok/s/GPU | vs VR200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 26.08.00 | nvfp4 | 4 | 64 | 2.211 | 2910.78 | **59,282** | **9.06x** |
| 26.08.00 | nvfp4 | 4 | 16 | 0.575 | 2797.83 | 56,988 | 8.71x |
| 26.08.00 | nvfp4 v2 | 4 | 16 | 0.579 | 2778.50 | 56,594 | 8.65x |
| 26.08.00 | nvfp4 | 2 | 8 | 0.317 | 2536.95 | 51,685 | 7.90x |
| 26.06.01 | nvfp4 | 4 | 16 | 0.647 | 2607.07 | 50,646 | 7.74x |
| 26.06.01 | nvfp4 | 2 | 8 | 0.353 | 2392.00 | 46,414 | 7.09x |
| 26.08.00 | fp8_cs | 4 | 16 | 0.810 | 1987.11 | 40,454 | 6.18x |
| 26.08.00 | fp8_cs | 2 | 8 | 0.420 | 1916.32 | 39,010 | 5.96x |
| 26.06.01 | fp8_cs | 4 | 16 | 0.860 | 1962.03 | 38,102 | 5.82x |
| 26.06.01 | fp8_cs | 2 | 8 | 0.453 | 1860.35 | 36,168 | 5.53x |
| 26.08.00 | bf16 fallback | 1 | 8 | 3.400 | 236.61 | 4,819 | VR200 +35.8% |
| 26.06.01 | bf16 fallback | 2 | 8 | 3.410 | 247.32 | 4,805 | VR200 +36.2% |
| 26.06.01 | bf16 fallback | 1 | 8 | 3.457 | 243.91 | 4,739 | VR200 +38.1% |

Deviations the quantized rows carry, to report alongside them:
`fp8_dot_product_attention=false` on every nvfp4 row (no TE fp8-attention
backend for sm_103), and `TP=PP=CP=1` pinned on every fp8_cs row (the stock
`gb300` fp8 preset is an 8-GPU shape with `TP*PP*CP=8`).

**Dense findings:**
- **nvfp4 > fp8_cs** by ~30-35% at every shape, on both containers.
- **MBS=4 is the ceiling.** MBS=8 OOMs at both precisions.
- **Gradient accumulation still pays** at the MBS ceiling: GA=1 -> 4 gave +4.0%.
  It had not plateaued, so GBS 128/256 remain untested.
- **Config variant v1 ~= v2** (v2 0.7% slower); not a meaningful lever.

### Qwen3 30B-A3B, full sweep (seq_len 4096, EP=4, alltoall)

| Container | Precision | MBS | GBS | s/iter | TFLOPS/GPU | tok/s/GPU | vs VR200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 26.06.01 | bf16 | 2 | 256 | 17.427 | 346.08 | **15,042** | **1.61x** |
| 26.08.00 | fp8_cs | 2 | 256 | 26.402 | 228.40 | 9,929 | 1.07x |
| 26.08.00 | bf16 | 1 | 256 | 32.335 | 186.50 | 8,107 | VR200 +14.9% |
| 26.06.01 | bf16 | 1 | 256 | 33.350 | 180.84 | 7,860 | VR200 +18.5% |
| 26.06.01 | bf16 fallback | 1 | 256 | 37.202 | 162.12 | 7,047 | VR200 +32.2% |
| 26.06.01 | fp8_cs | 1 | 256 | 53.072 | 113.64 | 4,939 | VR200 +88.6% |

**MoE findings — the opposite of the dense model:**

- **MBS dominates precision.** MBS=1 -> 2 at bf16, same container, nearly
  doubles throughput: 7,860 -> 15,042 tok/s/GPU (**1.91x**). No precision change
  came close to that.
- **fp8 *hurts* this MoE.** At MBS=1, same container, fp8_cs is **37% slower**
  than bf16 (4,939 vs 7,860) — slower even than the six-workaround bf16
  fallback. With `EP=4` the per-expert GEMMs are too small to amortise
  quantize/dequantize, so fp8 is pure overhead.
- **The workarounds barely matter here.** Dropping all six bought +11.5%
  (7,047 -> 7,860), versus 9.66x on the dense model, because at GBS=256/MBS=1
  the time goes into expert dispatch and small GEMMs rather than the fused
  attention and compile paths the workarounds disable.
- **Practical rule: size the microbatch first.** Enabling fp8 at low MBS on an
  MoE makes it slower.

## 3. What the missing kernels cost — measured, not extrapolated

Same node, same container, **identical batch shape** (MBS=2, GBS=8); the only
difference is the workaround set:

| llama3.1 8B, 26.06.01, MBS=2 GBS=8 | s/iter | tok/s/GPU |
| --- | --- | --- |
| bf16 fallback (six workarounds) | 3.410 | 4,805 |
| native nvfp4 (one workaround) | 0.353 | 46,414 |
| **cost of the fallback path** | | **9.66×** |

So essentially the entire GB300 advantage is *unlocked software*, not raw FLOPs.
Conversely, VR200 is leaving ~9.7× on the table purely because TE has no
`sm_107a` cubins and `ptxas` has no `sm_107a` target — on hardware that is
32–38% *faster* in the path it can run.

---

## 4. Container comparison, and why TFLOPS lies here

Same node, same config. `26.08.00` is off the recipe pin (`FW_VERSION=26.06.01`).

| Config | 26.06.01 s/iter | 26.08.00 s/iter | Wall-clock gain | TFLOPS change | FLOPs/iter ratio |
| --- | --- | --- | --- | --- | --- |
| bf16 fallback MBS=1 | 3.457 | 3.400 | **+1.7%** | −3.0% | 0.9541 |
| native nvfp4 MBS=2 | 0.353 | 0.317 | **+11.4%** | +6.1% | 0.9528 |

Two unrelated configs land on the same **0.953** ratio, so Megatron-Bridge's
model-FLOPs accounting changed ~4.6% between the images. The bf16 row is the
proof that this matters: `TFLOPS/GPU` says 26.08.00 got **3% worse** while the
stopwatch says it got **1.7% better**. Use `s/iter` or `tokens/s/GPU` across
containers.

`26.08.00` is worth having on GB300 (+1.7% to +11.4%) and does **nothing** for
sm_107 — its `ptxas` target list and TE cubin list are byte-identical to
`26.06.01`'s, with `te_ptx_entries=0` in both.

---

## 5. Arch-support probe, same two images on GB300

Confirms the container is not missing kernels for GB300 — only for sm_107.

| | 26.06.01 | 26.08.00 |
| --- | --- | --- |
| CUDA / TE / torch | 13.2.1 / 2.16.0 / 2.12.0a0 | 13.3.0 / 2.17.1 / 2.13.0a0 |
| `ptxas` accepts `sm_103a` | OK | OK |
| TE cubin archs | `sm_100 sm_100a sm_103a sm_120 sm_75 sm_80 sm_89 sm_90 sm_90a` | identical |
| `te_ptx_entries` | 0 | 0 |
| bf16 / fp8_cs / fp8_mx / nvfp4 | OK / OK / OK / OK | OK / OK / OK / OK |
| `torch.compile` | OK | OK |

The `ptxas` and TE cubin lists are **identical to those recorded on the VR200
machine**. `sm_103a` is present in both; `sm_107a` is in neither.

---

## 6. Validity checks

Worth recording because they are what make the comparisons above defensible.

| Check | Result |
| --- | --- |
| **Cross-node reproducibility** | Two GB300 nodes, same config: 3.457 vs 3.459 s/iter (**0.06%**), 243.91 vs 243.84 TFLOPS (0.03%). Cross-node container comparisons are valid. |
| **Repeatability** | Same config re-run on node 2: 0.317 s / 2537.96 → 0.317 s / 2536.95 (0.04%). |
| **Metric agreement** | On the apples-to-apples row, `tokens/s/GPU` reproduces the TFLOPS-derived +38.1% exactly. |
| **Config identity** | Qwen3 loss curve on GB300 matched the VR200 reference to 4 s.f. |
| **Order-of-magnitude** | The 1860 TFLOPS fp8 row ≈ 37% MFU against a ~5 PFLOPS B300 fp8 dense peak; the 2910 nvfp4 row ≈ lower MFU against roughly double the peak. Both self-consistent. |

---

## Pending

Both GB300 nodes were lost to a cluster-wide node failure mid-queue (idle nodes
600 -> 356, `down*` 123 -> 182 across the partition), so these gaps remain. All
completed results above were unaffected: the per-run logs live on shared
storage with metrics already parsed in place.

- **Qwen3 MBS=4** (both containers). MBS=1 -> 2 gave 1.91x, so this is the
  largest open question for the best MoE number. Untested, not known-bad — the
  one attempt died with the node.
- **Qwen3 fp8_cs at MBS=2 on 26.06.01**, the missing cell that would make the
  MBS=2 precision comparison same-container. The 26.08.00 fp8 row above cannot
  be compared directly against the 26.06.01 bf16 row (see section 4).
- **llama nvfp4 MBS=4 at GBS 128/256.** GA had not plateaued at GBS=64.
- `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` capture of TE's backend-selection
  reasoning for the fp8-attention failure.
- On the VR200 side, the arch gate and the `reference` row set are **done and
  reproduced** (see the re-measured table in section 1); `native` and
  `portable` were still running as of that commit. The recompute question in
  `portable` remains the largest potential win for a bf16-only machine — see
  [`RUNBOOK_VR200.md`](RUNBOOK_VR200.md).
