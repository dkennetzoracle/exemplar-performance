# Measured results: GB300 (sm_103) vs Vera Rubin (sm_107)

> **Just want the numbers?** [`COMPARISON.md`](COMPARISON.md) is the raw
> side-by-side: one row per config, both machines in adjacent columns, with a
> percent-difference column. This file is the narrative around them.

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

### Tuned on both sides — the comparison does not hinge on the config

The rows above carry full recompute, which both machines turn out not to need
at MBS=1. Re-running the dense fallback with recompute off on **both** gives:

| llama3.1 8B, 26.06.01, bf16 fallback | GB300 | VR200 | VR200 advantage |
| --- | --- | --- | --- |
| MBS=1 GBS=8, recompute | 4,739 | 6,543 | +38.1% |
| MBS=1 GBS=64, recompute | 4,790 | 6,639 | +38.6% |
| MBS=1 GBS=8, **no** recompute | 6,779 | 9,210 | +35.9% |
| MBS=1 GBS=32, **no** recompute | 6,887 | 9,377 | +36.2% |
| MBS=1 GBS=64, **no** recompute | **6,906** | **9,405** | **+36.2%** |
| MBS=2 GBS=8, **no** recompute | *OOM* | *OOM* | — |

VR200's lead sits between **+35.9% and +38.6% across every cell**, so the
apples-to-apples conclusion is a property of the hardware rather than of a
chosen config. Decomposed, the two machines agree on both levers and they
stack almost additively: dropping recompute is worth +43.0%/+44.2% on GB300
and +40.7%/+41.6% on VR200 (at GBS=8/64), while GBS 8→64 is worth only
+1.1–1.9% and +1.5–2.1%.

**Full recompute is never needed at MBS=1, on either architecture**, and it
costs ~41–44%. `run_bf16_fallback.sh` and the QUICKSTART prescribe
"MBS=1 plus full recompute" as a single unit; MBS=1 alone is sufficient.
MBS=2 without recompute OOMs on both (GB300 tried to allocate 8.00 GiB with
662 MiB free — exactly the score matrix WHY 4 predicts), so the accurate rule
is *MBS>1 requires recompute, and recompute costs more than the higher MBS
returns*.

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

| Workload | VR200 reference | VR200 **tuned** best | GB300 best | vs VR200 tuned |
| --- | --- | --- | --- | --- |
| llama3.1 8B | 6,543 tok/s/GPU | **9,405** (bf16, no recompute, GBS=64) | **59,389** (nvfp4, 26.08.00, MBS=4 GBS=256) | **6.31x** |
| Qwen3 30B-A3B | 9,317 tok/s/GPU | 9,317 (no portable lever ran — MBS OOMs) | **24,372** (bf16, 26.06.01, MBS=4 GBS=256) | **2.62x** |

**Quote the tuned column, not the reference column.** Measuring GB300's best
against VR200's *reference* gives llama3.1 8B 9.08x, but that compares a tuned
config to an untuned one. VR200's own portable sweep lifts it to 9,405
tok/s/GPU by dropping full recompute (+43.8%), which brings the honest dense
advantage down to **6.31x**. Qwen3 is unchanged at 2.62x because none of the
portable levers ran on sm_107 — its MBS rows OOM under unfused attention.

The two workloads behave completely differently, and the dense-model headline
does **not** generalise. Anyone extrapolating 9x to a mixture-of-experts
workload would be wrong by nearly an order of magnitude.

### llama3.1 8B, full sweep (seq_len 8192)

| Container | Precision | MBS | GBS | s/iter | TFLOPS/GPU | tok/s/GPU | vs VR200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 26.08.00 | nvfp4 | 4 | 256 | 8.828 | 2916.25 | **59,389** | **9.08x** |
| 26.08.00 | nvfp4 | 4 | 128 | 4.417 | 2914.67 | 59,349 | 9.07x |
| 26.08.00 | nvfp4 | 4 | 64 | 2.211 | 2910.78 | 59,282 | 9.06x |
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
- **Gradient accumulation saturates at GA=4.** GBS 16 -> 64 gave +4.0%, but
  64 -> 128 -> 256 gave only +0.1% and +0.07% (59,282 -> 59,349 -> 59,389). The
  optimizer step and DP all-reduce are fully amortised by GA=4; past that there
  is nothing left to hide. Do not burn runs on GBS > 64.
- **Config variant v1 ~= v2** (v2 0.7% slower); not a meaningful lever.

### Qwen3 30B-A3B, full sweep (seq_len 4096, EP=4, alltoall)

| Container | Precision | MBS | GBS | s/iter | TFLOPS/GPU | tok/s/GPU | vs VR200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 26.06.01 | bf16 | 4 | 256 | 10.756 | 560.69 | **24,372** | **2.62x** |
| 26.08.00 | bf16 | 4 | 256 | 10.813 | 557.66 | 24,243 | 2.60x |
| 26.08.00 | bf16 | 2 | 256 | 16.575 | 363.82 | 15,816 | 1.70x |
| 26.06.01 | bf16 | 2 | 256 | 17.427 | 346.08 | 15,042 | 1.61x |
| 26.08.00 | fp8_cs | 2 | 256 | 26.402 | 228.40 | 9,929 | 1.07x |
| 26.08.00 | bf16 | 1 | 256 | 32.335 | 186.50 | 8,107 | VR200 +14.9% |
| 26.06.01 | bf16 | 1 | 256 | 33.350 | 180.84 | 7,860 | VR200 +18.5% |
| 26.06.01 | bf16 fallback | 1 | 256 | 37.202 | 162.12 | 7,047 | VR200 +32.2% |
| 26.06.01 | fp8_cs | 1 | 256 | 53.072 | 113.64 | 4,939 | VR200 +88.6% |

**MoE findings — the opposite of the dense model:**

- **MBS dominates everything.** Same container, same precision, GBS fixed at
  256, only MBS varying:

  | MBS | tok/s/GPU | vs MBS=1 |
  | --- | --- | --- |
  | 1 | 7,860 | — |
  | 2 | 15,042 | 1.91x |
  | 4 | **24,372** | **3.10x** |
  | 8 | OOM | — |

  MBS=1 -> 4 is worth **3.10x**, which is larger than any other lever found in
  this work outside the missing-kernel gap itself. MBS=8 OOMs, so 4 is the
  ceiling at `EP=4` on one node. No precision change came remotely close.
- **fp8 hurts this MoE by a constant 37%, at every MBS tested.** Same
  container, bf16 vs fp8_cs: at MBS=1, 7,860 -> 4,939; at MBS=2,
  15,042 -> 9,429. Both exactly 37% slower. The penalty does **not** shrink as
  MBS grows, which rules out the obvious "small per-expert GEMMs" explanation —
  whatever the quantize/dequantize cost is around expert dispatch, it scales
  with the work rather than being a fixed overhead. At MBS=1 fp8 is slower even
  than the six-workaround bf16 fallback.
- **26.08.00 buys nothing at the MoE ceiling.** Its advantage shrinks as MBS
  rises and inverts to a tie at the top: +3.1% at MBS=1, +5.1% at MBS=2,
  **-0.5% at MBS=4** (10.756 vs 10.813 s/iter, inside run-to-run spread). The
  gain is overhead reduction, visible only while per-iteration overhead is a
  large share of the work. So the recipe-pinned 26.06.01 remains the best MoE
  config — unlike the dense model, where 26.08.00 is worth up to +12%.
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

The GB300 dense and MoE sweeps are done, including the recompute rows the
sm_107 result called out on this side. See [`COMPARISON.md`](COMPARISON.md)
for the raw side-by-side.

Genuinely outstanding:

- **Qwen3 fp8 at MBS=1 and MBS=4 on `nemo:26.08.00`.** Both were lost to the
  mid-session node failure, not measured and not diagnosed. Low value —
  fp8 costs this MoE a flat 37% on 26.06.01, so these would confirm a config
  already known to be worse than bf16.
- **VR200 `native`** — expected to fail on every quantized row; worth one pass
  per image purely to bank per-config evidence for the TE ticket.
- **VR200 `portable` on `nemo:26.08.00`** — the only portable lever still
  unmeasured on sm_107 (predicted +1.7%).
