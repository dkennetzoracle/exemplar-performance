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

"""Run an entrypoint with narrow compatibility patches applied first.

Only one patch today, for a library-internal bug that stops training after the
first logged iteration:

    File ".../megatron/bridge/training/utils/train_utils.py", training_log
        log_string += f" grad norm: {grad_norm:.3f} |"
    TypeError: unsupported format string passed to Tensor.__format__

Megatron-LM hands back `grad_norm` as a 0-dim tensor on some paths (it skips a
`.item()` to avoid a device sync); Megatron-Bridge's logger formats it with a
float spec. Both ship inside the same container image, so this is reachable on
any GPU, not just unproven ones. The upstream fix is a `float()` in
`training_log`.

Config-level avoidance was tried and does not hold: `optimizer.clip_grad=0.0`
plus `train.skip_sync_grad_norm_across_mp=true` still leaves a tensor, because
more than one code path can produce it. Patching the formatting instead fixes
the whole class of occurrence, and cannot change any computed value -- it only
affects how an already-computed scalar is rendered into a log string.

Usage (argv[1] onwards is the real entrypoint and its arguments):
    python compat_shim_local.py /path/to/run_script.py --flag ...
"""

from __future__ import annotations

import runpy
import sys


def _patch_tensor_format() -> None:
    """Let float format specs work on single-element tensors."""
    import torch

    original = torch.Tensor.__format__

    def __format__(self, format_spec):  # noqa: N807
        # Only intervene where the original raises: a non-empty spec (e.g.
        # ".3f") on a tensor holding exactly one value. Everything else,
        # including empty specs and real multi-element tensors, is untouched.
        if format_spec and self.numel() == 1:
            try:
                return format(self.item(), format_spec)
            except (TypeError, ValueError, RuntimeError):
                pass
        return original(self, format_spec)

    torch.Tensor.__format__ = __format__
    print(
        "compat_shim_local: patched torch.Tensor.__format__ "
        "(works around Tensor grad_norm vs float log format)",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <entrypoint.py> [args...]", file=sys.stderr)
        raise SystemExit(2)

    _patch_tensor_format()

    target = sys.argv[1]
    # Present argv to the target exactly as if it had been invoked directly.
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
