# Changing the run: batch shape, sequence length, length, and power capture

How to vary the knobs on the single-node Docker path, and how to run long
enough for someone to watch power while you do it.

Everything here is on top of the working launch in [`QUICKSTART.md`](QUICKSTART.md).
Nothing here changes what the perf matrix measured — those numbers came from the
committed defaults, and anything you change makes your run non-comparable to
them. That is fine for a power study; it is not fine for adding rows to
`perf_matrix/`.

---

## 0. Before anything: the credentials

`.ngc` and `.hf` are git-ignored and are **not** in the repo. On a fresh
machine you need both. On *this* machine, as of the last session:

* `nvcr.io` login **persists** in `~/.docker/config.json`, and both images
  (`nemo:26.06.01`, `nemo:26.08.00`) are already pulled — nothing to redo.
* The HF token is **gone**, but the Llama-3.1-8B tokenizer/config is already
  cached under `$LLMB_INSTALL/.cache/huggingface`. So you can run with

  ```bash
  export OFFLINE=true          # sets HF_HUB_OFFLINE=1
  export HF_TOKEN=unused       # only to satisfy the launcher's guard
  ```

  Put the real token back (`export HF_TOKEN=<token>`, drop `OFFLINE`) if you
  ever need a model whose config is not already cached.

---

## 1. The knobs, and where each one goes

Two different mechanisms, and picking the wrong one is the main way this goes
wrong:

| Knob | How | Notes |
| --- | --- | --- |
| **MBS** | `MBS=2` env var | first-class flag (`--micro_batch_size`) |
| **GBS** | `GBS=64` env var | first-class flag (`--global_batch_size`) |
| **steps** | `MAX_STEPS=500` env var | first-class flag (`--max_steps`) |
| **TP/PP/CP/VP/EP** | `TP=2` etc. env var | first-class flags |
| **seq_len** | `EXTRA_HYDRA_OVERRIDES="model.seq_length=4096"` | **no env var** — Hydra only |
| **recompute** | `EXTRA_HYDRA_OVERRIDES="model.recompute_granularity=null"` | Hydra only |
| **container** | `RUN_CONF_IMAGE=nvcr.io/nvidia/nemo:26.08.00` | default comes from the recipe's `launch.sh` |

`EXTRA_HYDRA_OVERRIDES` is a space-separated string appended to `run_script.py`'s
argument list. `run_bf16_fallback.sh` appends yours *after* its own, and later
Hydra overrides win, so you can override anything the fallback sets.

**Check before you commit to a long run.** `PRINT_ONLY=true` prints the
assembled `docker run` and exits without touching a GPU; `DRYRUN=true` actually
resolves the config inside the container and dumps
`configs/ConfigContainer.yaml`, then exits. `DRYRUN` is the one that catches a
bad Hydra key, and it costs about 90 seconds.

---

## 2. Change MBS / GBS

```bash
cd llama3.1/pretrain/megatron_bridge/single_node_docker
export LLMB_INSTALL=/mnt/nvme/llmb HF_TOKEN=unused OFFLINE=true

MBS=2 GBS=64 ./run_bf16_fallback.sh
```

**The one rule: `GBS` must be divisible by `MBS × DP`.** Here `DP = 4` (4 GPUs,
`TP=PP=CP=1`), so with `MBS=2`, `GBS` must be a multiple of 8. The remainder is
gradient accumulation:

```
GA = GBS / (MBS x DP)        tokens per iteration = GBS x seq_len
```

What to expect, from the measured matrix (`perf_matrix/SUMMARY.md`):

* **GBS up is nearly free but nearly pointless.** 8 → 64 bought +1.3–2.1%. It
  only amortises the optimizer step and the DP all-reduce over more
  micro-batches.
* **MBS up is worth more (+1.8–3.4%) but only while recompute is on.** With
  recompute *off*, `MBS=2` OOMs on both VR200 and GB300.
* **Recompute off is the big one: +40.7%.** See §5.

---

## 3. Change sequence length

**Set `model.seq_length` and nothing else.**

```bash
MBS=1 GBS=16 EXTRA_HYDRA_OVERRIDES="model.seq_length=4096" ./run_bf16_fallback.sh
```

Do **not** also set `dataset.seq_length`. It looks right — it is in the dumped
`ConfigContainer.yaml` — but that dump is post-resolution, and at override time
the key does not exist yet on 26.06.01:

```
omegaconf.errors.ConfigAttributeError: Key 'seq_length' is not in struct
hydra.errors.ConfigCompositionException: Could not override 'dataset.seq_length'.
```

The dataset length is *derived* from the model's. Verified on both containers
with `DRYRUN=true` after setting only `model.seq_length=4096`:

| container | dataset field | value |
| --- | --- | --- |
| 26.06.01 | `dataset.sequence_length` | 4096 |
| 26.08.00 | `dataset.seq_length` | 4096 |

Note the field is even *named* differently between the two, which is the other
reason not to set it by hand.

**Sequence length has an outsized memory effect on this path.** The bf16
fallback runs with `NVTE_UNFUSED_ATTN=1`, which materialises the full
`mbs × heads × seq × seq` score matrix — 8 GiB per tensor at `seq_len=8192`.
That is quadratic, so 8192 → 4096 cuts it 4×. Halving sequence length is the
cheapest way to buy headroom for a larger MBS on the fallback path.

Keep `GBS × seq_len` fixed if you want tokens-per-iteration held constant while
you vary the shape — e.g. `GBS=8` at 8192 and `GBS=16` at 4096 are both 65,536
tokens per iteration.

### Measured: halving sequence length nearly doubles throughput here

Run both sides of that pair and the difference is large. Same container
(26.06.01), same fallback, `MBS=1`, recompute on, identical 65,536 tokens per
iteration:

| seq_len | GBS | s/iter | TFLOPS/GPU | tok/s/GPU |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 8 | 2.504 | 336.81 | 6,543 |
| 4096 | 16 | 1.285 | 615.74 | **12,750** (+95%) |

That is the quadratic attention term, not a free win — **it is a different
training job.** The model now has half the context, so this number does not
belong anywhere near `perf_matrix/`, and it is not a like-for-like speedup. It
does mean that if you want a long, high-utilisation run to draw power against,
`seq_len=4096` gets you far more work done per minute.

---

## 4. Make the run longer

```bash
MAX_STEPS=500 ./run_bf16_fallback.sh
```

**Then re-parse with a later window, or your number describes the warm-up.**
The launcher's end-of-run parse always averages **iterations 35–44** — the
recipe's window, kept so numbers stay comparable to the Slurm path. On a
500-step run that is still measuring the first 9% of it.

```bash
# the launcher prints the log path; or find the newest:
LOG=$(ls -t /mnt/nvme/llmb/workloads/pretrain_llama3.1/experiments/*/*/*/log-*.out | head -1)

/mnt/nvme/llmb/parser-venv/bin/python parse_results_local.py "$LOG" \
  --min-iteration 100 --max-iteration 499
```

**Use the parser venv, not `python3`.** The parser imports
`llmb_run.pretrain_log_parser`, which needs `typer`. System python does not
have it, and the launcher's own end-of-run parse fails with

```
error: could not import llmb_run.pretrain_log_parser (No module named 'typer').
```

which is harmless — the run itself already finished and the log is on disk — but
it means you have to parse by hand afterwards.

Cost scales with the config, not just the step count. Measured on the fallback:
1.28 s/iter at `seq_len=4096 GBS=16`, ~2.5 s/iter at `seq_len=8192 GBS=8`. So
120 steps ≈ 2.5–5 min and 600 steps ≈ 13–25 min, plus ~4 min of container
startup and model init either way, plus ~20 s of post-training eval.

---

## 5. The knob actually worth turning: recompute

Full activation recompute is on by default in the fallback (it is what makes
`MBS=1` fit at all). Turning it off was worth **+40.7%** on VR200 — more than
every other portable lever combined:

```bash
EXTRA_HYDRA_OVERRIDES="model.recompute_granularity=null" \
  MBS=1 GBS=64 ./run_bf16_fallback.sh
```

`model.recompute_method` and `model.recompute_num_layers` do not need clearing;
`granularity=null` disables the feature. Keep `MBS=1` — `MBS=2` with recompute
off OOMs on both machines.

This is the highest-*throughput* configuration sm_107 can currently reach.
Whether it is also the highest-*power* one is untested, and not obvious:
recompute-on re-runs forward passes to rebuild activations, so it burns extra
FLOPs for no extra tokens. It may well draw more watts while delivering 40%
less useful work. If your teammate cares about performance-per-watt rather than
peak watts, run it both ways — it is one flag and two runs.

---

## 6. Power metrics

[`collect_power.sh`](collect_power.sh) samples per-GPU power, clocks, thermals
and utilisation from NVML on the **host** and writes a CSV.

```bash
# sample for the duration of the run, then print a summary
./collect_power.sh -o power.csv -- ./run_bf16_fallback.sh

# or sample independently, in another terminal, Ctrl-C to stop
./collect_power.sh -o power.csv -i 1

# summarise a CSV you already have
./collect_power.sh --summarize power.csv
```

Options: `-o FILE`, `-i SECS` (default 1), `-g 0,1` (subset of GPUs).

### Why not `ENABLE_GPU_METRICS=true`

That flag already exists, but it routes through Nsight (`NSYS_GPU_METRICS=1`)
and is the wrong tool here. It only samples inside the profile step window
(`PROFILE_START_STEP`/`PROFILE_STOP_STEP`, default 45–50), it writes into a
`.nsys-rep` your teammate then has to open in the Nsight GUI, and `nsys` perturbs
the timings. `collect_power.sh` covers the whole run, lands in a CSV anything
can plot, and costs the run nothing.

Use `ENABLE_GPU_METRICS=true` when you want power correlated with *kernels* on a
timeline. Use `collect_power.sh` when you want power over *time*.

### Recommended power run

Long enough to reach steady state, in the highest-throughput config sm_107 can
run (see §5 on why that is not necessarily the highest-wattage one):

```bash
cd llama3.1/pretrain/megatron_bridge/single_node_docker
export LLMB_INSTALL=/mnt/nvme/llmb HF_TOKEN=unused OFFLINE=true

MAX_STEPS=600 MBS=1 GBS=8 \
EXTRA_HYDRA_OVERRIDES="model.recompute_granularity=null" \
./collect_power.sh -o ~/power-$(date +%F).csv -i 1 -- ./run_bf16_fallback.sh
```

≈ 18 min of training (1.78 s/iter) plus ~4 min of startup. **Watch GBS when you
size the run** — `s/iter` scales with it, so the same 600 steps at `GBS=64`
is 13.9 s/iter, i.e. **2.3 hours**, not 18 minutes. Raise `MAX_STEPS` for a
longer soak; raise `GBS` only if you want fewer, larger iterations.

Then re-parse the throughput over a steady-state window:

```bash
LOG=$(ls -t /mnt/nvme/llmb/workloads/pretrain_llama3.1/experiments/*/*/*/log-*.out | head -1)
/mnt/nvme/llmb/parser-venv/bin/python parse_results_local.py "$LOG" \
  --min-iteration 100 --max-iteration 599
```

### Measured: what a run actually looks like

From the validation run behind this guide — `seq_len=4096 GBS=16 MBS=1`,
recompute on, 120 steps, 1 s sampling, 1,032 samples over 265 s:

```
GPU    mean W    max W    p95 W  mean SM MHz  max degC  mean util%
  0    1279.7   1578.2   1566.4         2103        47        77.5
  1    1306.8   1623.4   1608.5         2099        48        76.8
  2    1318.2   1630.7   1618.8         2103        49        76.7
  3    1308.6   1617.2   1599.9         2108        47        76.5
all    5213.4 (sum of per-GPU means, W)

energy over the sampled span: 0.3833 kWh
```

Two things to read off that. The **mean is well below the max** (1,300 vs
1,600 W) because the mean includes container startup and model init, when the
GPUs are idle — for a training-power figure use p95, or slice the CSV to the
training window by timestamp. And **1,600 W peak against a 2,300 W limit at 77%
utilisation** says this config is not power-limited; it is limited by the
missing kernels, exactly as `perf_matrix/SUMMARY.md` describes.

### Reading the CSV

Columns: `index, timestamp, power.draw [W], power.limit [W],
clocks.current.sm [MHz], clocks.current.memory [MHz], temperature.gpu,
temperature.memory, utilization.gpu [%], utilization.memory [%],
memory.used [MiB], pstate, clocks_event_reasons.active`.

One row per GPU per sample, so a 4-GPU run at 1 s gives 4 rows/second. Note
nvidia-smi keeps the unit in the header even under `nounits`, pads values with a
leading space, and renamed `clocks_throttle_reasons` → `clocks_event_reasons` —
strip/normalise the headers before loading into pandas.

Three things worth telling your teammate up front:

* **Idle is not zero, and idle is not one number.** Against a **2300 W** limit
  these GPUs draw **265–277 W each cold** (node idle ~1.09 kW, measured after
  10 days quiet) but **388–410 W warm** (~1.6 kW, measured minutes after a
  run). That is a 47% spread on the baseline you are about to subtract, so
  **capture your own baseline immediately before the run** rather than reusing
  a number from this doc:

  ```bash
  ./collect_power.sh -o idle.csv -i 1 -- bash -c 'sleep 60'
  ```
* **The first iterations are not steady state.** Power ramps and clocks settle
  over the early iterations, which is why the run should be long and why the
  throughput window should skip the start.
* **This is a bring-up config, not a benchmark.** bf16 with unfused attention on
  4 GPUs is not what this silicon will draw in a real nvfp4 run — see
  `perf_matrix/SUMMARY.md`. It is a floor, not a representative load.
  `clocks_event_reasons.active` will tell you whether you are power-, thermal-
  or utilisation-limited; on this config expect utilisation-limited.

DCGM is also installed on this node (`dcgmi discovery -l` sees all 4 GPUs) if
your teammate wants field-group profiling instead of NVML polling.

---

## 7. Driving power up: measured

The default bring-up config is a poor power load. Measured on 4x sm_107
(2300 W cap each), 1s sampling unless noted, container 26.08.00:

| config | TFLOPS/GPU | s/iter | max W/GPU | p95 W | node mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| seq 8192, MBS 1, GBS 8, recompute **off** | 465 | 1.73 | 1328-1358* | - | 3,864 W |
| seq 2048, MBS 8, GBS 64, recompute on | 918 | 1.58 | 1921-1967 | 1913-1960 | 5,099 W |
| **seq 2048, MBS 16, GBS 128, recompute on** | **934** | 3.11 | **1942-1989** | 1925-1982 | **5,877 W** |
| seq 2048, MBS 32, GBS 128, recompute on | 934 | 3.11 | 1924-1971 | 1912-1954 | 5,203 W |

\* 15s sampling. That does **not** understate the peak: the same DCGM load
sampled at 1s and at 15s peaked at 2293-2302 W and 2293-2309 W respectively, so
a sustained load's maximum is captured either way. 15-30s is fine; the mean is
the figure to distrust, because it includes container startup and model init.

**Power tracks compute density, i.e. TFLOPS/GPU.** Doubling achieved TFLOPS
(465 -> 934) lifted peak power ~50% (1,328 -> 1,989 W), to 86% of the cap. Three
things get you there, and the first is by far the biggest:

* **Shorter sequences.** The fallback runs `NVTE_UNFUSED_ATTN=1`, which
  materialises the score matrix -- O(seq^2) and memory-bound. Every second spent
  there is a second the tensor cores are idle. 8192 -> 2048 is 16x less attention
  work per sequence, so the GEMMs dominate.
* **Bigger MBS, which the shorter sequence pays for.** Larger GEMMs, better
  tensor-core occupancy. MBS 16 is the plateau: MBS 32 measured *identical*
  934 TFLOPS / 3.11 s/iter, because at GBS 128 with DP 4 it reduces to GA=1 and
  the micro-batch shape stops changing.
* **Recompute ON, not off.** Backwards from the throughput advice in section 5.
  Recompute re-runs forward passes, so it burns extra FLOPs for no extra tokens
  -- exactly what a power load wants, and it buys the memory headroom for the
  larger MBS. The framework's TFLOPS counter reports *model* FLOPs, so the real
  executed total is higher still.

**Why it stops at ~86%.** Two ceilings are not tunable from here: there is no
fp8/nvfp4 tensor-core path (TE ships no sm_107 cubins) and no fused attention
(cuDNN builds no sm_107 plan). DCGM's `targeted_power` reaches the cap with
DGEMM, which training cannot use. Closing the rest needs kernels in the
container -- see `perf_matrix/ISSUES.md`.

### Power-maximising training command

```bash
cd llama3.1/pretrain/megatron_bridge/single_node_docker
export LLMB_INSTALL=/mnt/nvme/llmb HF_TOKEN=unused OFFLINE=true

MAX_STEPS=580 MBS=16 GBS=128 \
RUN_CONF_IMAGE=nvcr.io/nvidia/nemo:26.08.00 \
EXTRA_HYDRA_OVERRIDES="model.seq_length=2048" \
./collect_power.sh -o ~/power-training-$(date +%F).csv -i 30 -- ./run_bf16_fallback.sh
```

30 min at 3.11 s/iter. Note there is no `recompute_granularity=null` here --
leaving the fallback's recompute in place is deliberate.

## 8. Inference power, and why FP8 lost

Same node, TP4, `vr_inference_benchmarks` via `run-sweep.sh --sweep saturate`:

| model | precision | max W/GPU | node mean |
| --- | --- | ---: | ---: |
| Qwen3.6-27B, dense | BF16 | 2031-2068* | 5,241 W |
| DeepSeek-V4-Flash, MoE | FP8 + FP8 KV | 1428-1465 | 3,736 W |

\* 15s sampling. Verified not to matter for a peak on a sustained load (see
section 7), and the 600 W gap is far outside any sampling artefact.

FP8 on the bigger model draws **less**, not more. DeepSeek-V4-Flash activates
10B of 290.9B parameters per token, so it is bandwidth-bound fetching expert
weights rather than compute-bound, and FP8 shrinks the bytes per MAC without
removing the bottleneck. For a power ceiling, **arithmetic density beats
precision and beats parameter count**: pick the dense model.

The repo's own numbers go further -- `saturate.sh:8-28` measured 4x TP1 Qwen at
2095 W/GPU against 1850 W/GPU for 1x TP4, because 4 independent replicas have
no tensor-parallel collectives to stall on. If the goal is purely maximum
inference power rather than a TP4 datapoint, `saturate.sh` unmodified is the
stronger load.

## 9. Sequencing the three power tests

Run them one at a time -- DCGM diag needs the GPUs to itself, and they cannot
share anyway. Put the inference run **last**:

`run-sweep.sh` leaks its vLLM server. `vllm_launch` bare-metal runs
`"$VLLM_BIN" "$@"` inside a shell function, so `SERVER_PID=$!`
(`run-sweep.sh:88`) captures the function's *subshell*, not the server.
Teardown kills the subshell and the real server is reparented to init, still
holding ~257 GB per GPU. The next job then dies with a misleading OOM --
"11.32 GiB is free" while the process itself only asked for 13.49 GiB. The
docker path kills by container name and is immune, which is why GB300 never
sees it. After any inference run:

```bash
pkill -f 'VLLM::'; pkill -f 'venv_vllm/bin/vllm'; sleep 10
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Also note the training job exits **1** even on a fully successful run: a
`CUDACachingAllocator.cpp:3113` assert fires during shutdown after the last
iteration. Check the iteration count, not the exit code.
