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

"""Report s/iter and TFLOPS/GPU for a local (non-Slurm) training log.

``llmb-run jobs`` cannot be used here: it refreshes Slurm state and looks up
logs by Slurm job id. The parsing itself is Slurm-independent, so this reuses
``llmb_run.pretrain_log_parser`` from the repo -- the same averaging window
(iterations 35-44) and the same NaN-grad-norm rejection the Slurm path applies,
so numbers stay directly comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FRAMEWORK = "megatron_bridge"


def _load_parser():
    """Import the repo's pretrain log parser without needing llmb-run installed."""
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "cli" / "llmb-run" / "src"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
    try:
        from llmb_run.pretrain_log_parser import PretrainLogParseStatus, parse_pretrain_log
    except ImportError as exc:  # pragma: no cover - depends on checkout layout
        raise SystemExit(
            f"error: could not import llmb_run.pretrain_log_parser ({exc}).\n"
            f"       Expected it under {candidate}."
        ) from exc
    return parse_pretrain_log, PretrainLogParseStatus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="Path to log-<experiment_name>.out")
    ap.add_argument(
        "--min-iteration",
        type=int,
        default=None,
        help="First iteration to average (default: the recipe's 35).",
    )
    ap.add_argument(
        "--max-iteration",
        type=int,
        default=None,
        help="Last iteration to average (default: the recipe's 44).",
    )
    args = ap.parse_args()

    if not args.log.is_file():
        print(f"error: no such log file: {args.log}", file=sys.stderr)
        return 2

    parse_pretrain_log, Status = _load_parser()

    kwargs = {}
    if args.min_iteration is not None:
        kwargs["min_iteration"] = args.min_iteration
    if args.max_iteration is not None:
        kwargs["max_iteration"] = args.max_iteration

    result = parse_pretrain_log(args.log, FRAMEWORK, **kwargs)

    print()
    print("=" * 67)
    print(" Results")
    print("=" * 67)

    if result.status == Status.SUCCESS:
        m = result.metrics
        print(f" Averaging window:   iterations {result.min_iteration}-{result.max_iteration}")
        print(f" Samples:            {m.time_sample_count}")
        print(f" s/iter:             {m.time_mean_seconds:.3f} (std {m.time_std_seconds:.3f})")
        if m.tflops_per_gpu_mean is not None:
            print(f" TFLOPS/GPU:         {m.tflops_per_gpu_mean:.2f} (std {m.tflops_per_gpu_std:.2f})")
        else:
            print(" TFLOPS/GPU:         not reported in this log")
        print("=" * 67)
        return 0

    if result.status == Status.INVALID_GRAD_NORM:
        print(f" FAILED: NaN grad norm first seen at iteration {result.invalid_grad_norm_iteration}.")
        print(" The run is numerically broken; timings are not meaningful.")
    elif result.status == Status.INCOMPLETE:
        print(
            f" INCOMPLETE: log reached iteration {result.max_iteration_seen}, "
            f"need {result.max_iteration}."
        )
        print(" Re-run with MAX_STEPS >= 50, or pass --min-iteration/--max-iteration")
        print(" to average a shorter window (not comparable to published numbers).")
    elif result.status == Status.NO_DATA:
        print(" NO DATA: no per-iteration timing lines found.")
        print(" Check the log for a startup failure.")
    else:
        print(f" {result.status.value.upper()}")

    print("=" * 67)
    return 1


if __name__ == "__main__":
    sys.exit(main())
