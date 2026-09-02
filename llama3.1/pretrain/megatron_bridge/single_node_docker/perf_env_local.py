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

"""Resolve the performance environment that the Slurm path applies via Nemo-Run.

On Slurm, two pieces of Megatron-Bridge cooperate to build the container
environment, and neither is reachable without a ``SlurmExecutor``:

* ``scripts/performance/utils/executors.py`` -- ``PERF_ENV_VARS`` plus the
  GPU/HF-token/offline adjustments made in ``slurm_executor()``.
* ``scripts/performance/perf_plugins.py`` -- ``PerfEnvPlugin``, which mutates
  ``executor.env_vars`` and appends Hydra overrides to the run script's argv.

This module reproduces those *rules* for a single-node Docker launch. The
*data* they branch on (parallelism, batch sizes, CUDA-graph mode, MoE backend,
FSDP/NCCL-UB flags) is not copied: it is imported live from the pinned
Megatron-Bridge checkout via ``utils.utils.get_workload_base_config``, which is
pure stdlib and needs neither ``megatron.bridge`` nor ``nemo_run``. So a config
change upstream is picked up automatically; only a change to the env *rules*
themselves would need mirroring here.

Emits shell-safe ``KEY=VALUE`` lines for ``docker run --env-file`` (``--emit
env``) or whitespace-separated Hydra overrides for the run script (``--emit
hydra``).

Cross-checked against Megatron-Bridge commit
b50da4c7404caa41793e74ac40d18798844c7b67.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mirrored from scripts/performance/utils/executors.py::PERF_ENV_VARS
# ---------------------------------------------------------------------------
PERF_ENV_VARS = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
    "TRANSFORMERS_OFFLINE": "1",  # Default for benchmark runs that mostly use NullTokenizer.
    "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
    "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
    "NVTE_NORM_FWD_USE_CUDNN": "1",
    "NVTE_NORM_BWD_USE_CUDNN": "1",
    "TORCH_NCCL_HIGH_PRIORITY": "1",
    "HF_HUB_OFFLINE": "0",  # Keep HF Hub online by default; --offline flips this to 1.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_GRAPH_REGISTER": "0",
}

# scripts/performance/perf_plugins.py::PerfEnvPlugin._set_num_cuda_device_max_connections
SM100_OR_NEWER = ["b300", "b200", "gb200", "gb300", "vr200"]

# scripts/performance/utils/executors.py::slurm_executor (numa_cmd)
NUMA_DIVISOR_4GPU = ["gb200", "gb300", "vr200"]

# PerfEnvPlugin dataclass defaults that the recipe never overrides.
ENABLE_LAYERNORM_SM_MARGIN = True
ENABLE_MANUAL_GC = True
MANUAL_GC_INTERVAL = 100

logger = logging.getLogger("perf_env_local")


def _import_upstream(mbridge_dir: Path):
    """Import the config helpers from the pinned Megatron-Bridge checkout."""
    perf_dir = mbridge_dir / "scripts" / "performance"
    if not (perf_dir / "utils" / "utils.py").is_file():
        raise SystemExit(
            f"error: {perf_dir}/utils/utils.py not found.\n"
            f"       --mbridge-dir must point at a Megatron-Bridge checkout "
            f"(run setup_local.sh first)."
        )
    # configs.<family> and utils.* are imported by bare name upstream, so the
    # performance dir has to be on sys.path exactly as PYTHONPATH does it there.
    sys.path.insert(0, str(perf_dir))
    from utils.utils import get_exp_name_config, get_workload_base_config  # noqa: PLC0415

    return get_workload_base_config, get_exp_name_config


def resolve_env(args, cfg) -> dict[str, str]:
    """Build the container env for one workload, mirroring the Slurm path."""
    gpu = args.gpu
    family = args.model_family_name
    recipe = args.model_recipe_name
    task = args.task
    dtype = args.compute_dtype

    # --- utils/executors.py::slurm_executor -------------------------------
    env = PERF_ENV_VARS.copy()

    if gpu == "gb200":
        env["NCCL_NET_GDR_LEVEL"] = "PHB"  # For NCCL 2.25
        env["NCCL_NET_GDR_C2C"] = "1"  # For NCCL 2.26

    if args.nemo_home:
        env["NEMO_HOME"] = args.nemo_home

    if args.hf_token_in_env:
        # The token itself is passed through `docker run -e HF_TOKEN` so it is
        # never written to the run directory; only the side effect is recorded.
        env["TRANSFORMERS_OFFLINE"] = "0"
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"

    # --- setup_experiment.py (custom_env_vars for NCCL user buffers) ------
    if cfg.nccl_ub:
        env["NCCL_NVLS_ENABLE"] = "1"
        env["NCCL_CTA_POLICY"] = "1"

    # --- perf_plugins.py::PerfEnvPlugin.setup -----------------------------
    tp = cfg.tensor_model_parallel_size
    pp = cfg.pipeline_model_parallel_size
    cp = cfg.context_parallel_size
    # A preset can select a dispatcher the machine cannot use (HybridEP assumes
    # an NVL72 domain, so it is wrong on a single node). When the caller
    # overrides it, the env has to follow, or we would export MNNVL settings for
    # a dispatcher that is not running.
    moe_backend = args.moe_backend or getattr(cfg, "moe_flex_dispatcher_backend", None)
    moe_a2a_overlap = bool(getattr(cfg, "moe_a2a_overlap", False))

    # _set_num_cuda_device_max_connections
    max_connections = 8
    if moe_backend in ("deepep", "hybridep"):
        max_connections = 32
    if gpu in SM100_OR_NEWER:
        # Extra connections avoid serialization of streams on Blackwell+.
        max_connections = 32
    elif (tp > 1 or cp > 1) and not moe_a2a_overlap:
        # Force program-order kernel launch so comms overlap the GEMM.
        max_connections = 1
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = str(max_connections)

    # _set_layernorm_sm_margin
    if ENABLE_LAYERNORM_SM_MARGIN:
        margin = "20" if moe_backend in ("deepep", "hybridep") else "16"
        env["NVTE_FWD_LAYERNORM_SM_MARGIN"] = margin
        env["NVTE_BWD_LAYERNORM_SM_MARGIN"] = margin

    # _set_nvl_domain_size (HybridEP only)
    if moe_backend == "hybridep":
        # Effective EP, not the preset's: a caller overriding EP to fit the node
        # must get matching NVL-domain settings.
        ep = _effective(args, cfg, "expert_model_parallel_size")
        if gpu in ("h100", "b200", "b300"):
            env["NVLINK_DOMAIN_SIZE"] = "8"
            env["USE_MNNVL"] = "0"
            env["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = "8" if ep > 8 else str(ep)
        else:
            if ep > 72:
                raise SystemExit("error: ep_size must be less than or equal to 72")
            env["NVLINK_DOMAIN_SIZE"] = "72"
            env["USE_MNNVL"] = "1"
            env["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = str(ep)
        env["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] = "128"

    # _set_nccl_pp_comm_chunksize
    if recipe in ("llama3_70b", "llama31_405b") and task == "pretrain":
        chunksize = 2097152
    elif family == "llama" and task == "sft":
        chunksize = 2097152
    else:
        chunksize = None
    if pp > 1 and chunksize is not None:
        env["NCCL_P2P_NET_CHUNKSIZE"] = str(chunksize)

    # _set_model_specific_environment_variables
    if family == "deepseek":
        env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    cuda_graph_impl = cfg.cuda_graph_impl
    cuda_graph_scope = cfg.cuda_graph_scope or []
    if cuda_graph_impl == "full_iteration" or (
        cuda_graph_impl == "local" and "full_iteration" in cuda_graph_scope
    ):
        current = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "graph_capture_record_stream_reuse" not in current:
            sep = "," if current else ""
            env["PYTORCH_CUDA_ALLOC_CONF"] = f"{current}{sep}graph_capture_record_stream_reuse:True"
        env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "0"

    if getattr(cfg, "cutedsl_fused_grouped_mlp", False):
        env["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
        if moe_a2a_overlap:
            env["CUDNNFE_CLUSTER_OVERLAP_MARGIN"] = "8"

    if cfg.nccl_ub is True or (family == "llama" and cfg.use_megatron_fsdp is True):
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "NCCL_GRAPH_REGISTER"):
            env.pop(key, None)

    del_cudnn_ln = True
    if gpu == "h100":
        if family == "llama" and recipe == "llama3_8b" and task == "pretrain":
            if dtype == "fp8_cs":
                env["NCCL_CTA_POLICY"] = "1"
                del_cudnn_ln = False
    if gpu in ("gb200", "gb300", "vr200"):
        if family == "llama" and recipe == "llama2_70b":
            del_cudnn_ln = False
        if family == "llama" and recipe == "llama3_70b" and task == "pretrain":
            if dtype in ("bf16", "fp8_cs"):
                del_cudnn_ln = False
        if family == "llama" and recipe == "llama31_405b" and task == "pretrain":
            if dtype == "fp8_cs":
                del_cudnn_ln = False
        if family in ("deepseek", "kimi") and dtype == "fp8_mx":
            del_cudnn_ln = False
        if family == "gpt_oss" and recipe == "gpt_oss_20b" and task == "pretrain":
            del_cudnn_ln = False
    if family == "llama" and task == "sft":
        del_cudnn_ln = False
    if recipe == "nemotron_3_nano":
        del_cudnn_ln = False
    if del_cudnn_ln:
        env.pop("NVTE_NORM_FWD_USE_CUDNN", None)
        env.pop("NVTE_NORM_BWD_USE_CUDNN", None)

    if gpu == "b300":
        env["NCCL_IGNORE_CPU_AFFINITY"] = "1"

    if dtype == "nvfp4":
        env["NVTE_USE_FAST_MATH"] = "1"

    if args.deterministic:
        env["NCCL_ALGO"] = "Ring"
        env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return env


def resolve_hydra_overrides(args, cfg) -> list[str]:
    """Hydra overrides that PerfEnvPlugin appends to the run script's argv."""
    del cfg  # only the plugin defaults matter today
    if not ENABLE_MANUAL_GC:
        return []
    # perf_plugins.py::_default_perf_env_converter
    return [
        f"train.manual_gc={str(ENABLE_MANUAL_GC).lower()}",
        f"train.manual_gc_interval={MANUAL_GC_INTERVAL}",
    ]


def _effective(args, cfg, name: str):
    """Value of one parallelism/batch knob after applying a CLI override."""
    override = getattr(args, name)
    if name == "virtual_pipeline_model_parallel_size":
        # -1 is upstream's "not specified"; None is a meaningful override.
        return getattr(cfg, name) if override == -1 else override
    return getattr(cfg, name) if override is None else override


def resolve_plan(args, cfg) -> list[str]:
    """Resolved parallelism and batch sizing, as shell-eval-able KEY=VALUE pairs."""
    tp = _effective(args, cfg, "tensor_model_parallel_size")
    pp = _effective(args, cfg, "pipeline_model_parallel_size")
    cp = _effective(args, cfg, "context_parallel_size")
    vp = _effective(args, cfg, "virtual_pipeline_model_parallel_size")
    ep = _effective(args, cfg, "expert_model_parallel_size")
    mbs = _effective(args, cfg, "micro_batch_size")
    num_gpus = args.num_gpus

    if num_gpus % (tp * pp * cp) != 0:
        raise SystemExit(
            f"error: num_gpus={num_gpus} is not divisible by TP*PP*CP={tp * pp * cp}; "
            f"this workload cannot be sharded over {num_gpus} GPUs."
        )
    dp = num_gpus // (tp * pp * cp)

    # utils/overrides.py::set_post_overrides -- GBS follows weak scaling unless
    # the caller pinned it.
    if args.global_batch_size is not None:
        gbs = args.global_batch_size
    elif num_gpus != cfg.num_gpus:
        gbs = int(cfg.gbs_scaling_factor * num_gpus)
    else:
        gbs = cfg.global_batch_size

    if gbs < 1:
        # Presets tuned for large scales have gbs_scaling_factor < 1, so weak
        # scaling floors to zero on a single node.
        raise SystemExit(
            f"error: weak scaling this preset to {num_gpus} GPUs yields global "
            f"batch size {gbs}. Pin one explicitly with GBS=<n> "
            f"(a multiple of DP*MBS={dp * mbs})."
        )

    if gbs % (dp * mbs) != 0:
        raise SystemExit(
            f"error: global batch size {gbs} is not divisible by DP*MBS={dp * mbs}. "
            f"Pin a compatible GBS (env GBS=...) or MBS (env MBS=...)."
        )

    return [
        f"TP={tp}",
        f"PP={pp}",
        f"CP={cp}",
        f"VP={vp}",
        f"EP={ep}",
        f"DP={dp}",
        f"MBS={mbs}",
        f"GBS={gbs}",
        f"GA={gbs // (dp * mbs)}",
        f"CG={cfg.cuda_graph_impl}",
        f"FSDP={bool(cfg.use_megatron_fsdp)}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mbridge-dir", type=Path, required=True)
    parser.add_argument("--model_family_name", required=True)
    parser.add_argument("--model_recipe_name", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--compute_dtype", required=True)
    parser.add_argument("--task", default="pretrain")
    parser.add_argument("--config_variant", default="v2")
    parser.add_argument("--num_gpus", type=int, required=True)
    # Same names/semantics as scripts/performance/argument_parser.py so the
    # resolved plan and experiment name match what run_script.py will do.
    parser.add_argument("--tensor_model_parallel_size", type=int)
    parser.add_argument("--pipeline_model_parallel_size", type=int)
    parser.add_argument("--context_parallel_size", type=int)
    parser.add_argument("--virtual_pipeline_model_parallel_size", type=int, default=-1)
    parser.add_argument("--expert_model_parallel_size", type=int)
    parser.add_argument("--expert_tensor_parallel_size", type=int)
    parser.add_argument("--global_batch_size", type=int)
    parser.add_argument("--micro_batch_size", type=int)
    parser.add_argument("--nemo_home", default="")
    parser.add_argument(
        "--moe-backend",
        default="",
        help="Override the preset's MoE flex dispatcher for env purposes "
        "(deepep/hybridep enable the NVL-domain settings; anything else, e.g. "
        "alltoall, disables them).",
    )
    parser.add_argument(
        "--hf-token-in-env",
        action="store_true",
        help="HF_TOKEN is present in the launching shell (flips TRANSFORMERS_OFFLINE off).",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--emit",
        choices=["all", "env", "hydra", "plan", "exp-name", "numa-divisor"],
        default="all",
        help="all: shell-eval-able EXP_NAME/NUMA_DIVISOR/PLAN/HYDRA (and the env "
        "file, if --env-file is given); env: docker --env-file lines; hydra: "
        "run_script.py overrides; plan: resolved parallelism as KEY=VALUE; "
        "exp-name: Slurm-compatible experiment name; numa-divisor: ranks per "
        "NUMA node.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="With --emit all, write the docker env file here.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Upstream logs its config lookup at INFO on stdout; keep stdout clean so
    # the caller can consume it directly.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    if args.emit == "numa-divisor":
        print(2 if args.gpu in NUMA_DIVISOR_4GPU else 4)
        return

    get_workload_base_config, get_exp_name_config = _import_upstream(args.mbridge_dir)
    cfg = get_workload_base_config(
        args.model_family_name,
        args.model_recipe_name,
        args.gpu,
        args.compute_dtype,
        args.task,
        args.config_variant,
    )

    def env_lines() -> list[str]:
        lines = []
        for key, value in sorted(resolve_env(args, cfg).items()):
            if "\n" in value:
                raise SystemExit(f"error: refusing to emit multi-line value for {key}")
            lines.append(f"{key}={value}")
        return lines

    def exp_name() -> str:
        # Reuse upstream's own name builder so local runs land in the same
        # experiments/<name>/ layout that llmb-run and the recipe README expect.
        exp_config = get_exp_name_config(
            args,
            args.model_family_name,
            args.model_recipe_name,
            args.gpu,
            args.compute_dtype,
            args.task,
            args.config_variant,
        )
        return f"{args.task}_{args.model_recipe_name}_{args.compute_dtype}_{exp_config}"

    if args.emit == "env":
        print("\n".join(env_lines()))
    elif args.emit == "hydra":
        print(" ".join(resolve_hydra_overrides(args, cfg)))
    elif args.emit == "plan":
        print(" ".join(resolve_plan(args, cfg)))
    elif args.emit == "exp-name":
        print(exp_name())
    elif args.emit == "all":
        # One process, one config lookup, one copy of any upstream warning.
        if args.env_file is not None:
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text("\n".join(env_lines()) + "\n")
        values = {
            "EXP_NAME": exp_name(),
            "NUMA_DIVISOR": str(2 if args.gpu in NUMA_DIVISOR_4GPU else 4),
            "PLAN": " ".join(resolve_plan(args, cfg)),
            "HYDRA": " ".join(resolve_hydra_overrides(args, cfg)),
        }
        for key, value in values.items():
            print(f"{key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
