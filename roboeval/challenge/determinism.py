"""Make rollouts bit-reproducible across x86-64 machines.

NumPy's BLAS (OpenBLAS in the pip wheel) picks CPU-specific kernels and thread
counts at load time, and its rounding feeds back into the simulation through
IK and pose math: the same seed and actions can give a different episode on a
different CPU. Pinning one kernel and one thread for every evaluator, validator,
and the portal makes logged actions replay exactly anywhere.

The variables only take effect before NumPy is loaded, so command-line entry
points call ``ensure_pinned()`` first, which re-executes the process if needed.
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict, Optional, Sequence

# Haswell (AVX2) kernels run on any x86-64 CPU from 2013 on, Intel or AMD.
PINNED_ENV = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
}


def is_pinned() -> bool:
    return all(os.environ.get(k) == v for k, v in PINNED_ENV.items())


def _cpu_has_avx2() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            return " avx2" in f.read()
    except OSError:
        return False


def ensure_pinned(module: str, argv: Sequence[str]) -> None:
    """Re-run ``python -m <module> <argv>`` with the pinned math environment."""
    if is_pinned():
        return
    if platform.machine() not in ("x86_64", "AMD64") or not _cpu_has_avx2():
        sys.stderr.write(
            "warning: reproducible replay needs an x86-64 CPU with AVX2; results from this machine "
            "may not replay exactly elsewhere and can be held for organizer review.\n")
        return
    env = {**os.environ, **PINNED_ENV}
    os.execve(sys.executable, [sys.executable, "-m", module, *argv], env)


def math_fingerprint() -> Dict[str, Any]:
    """What determines floating-point results outside MuJoCo; stored in provenance."""
    import numpy as np

    blas: Optional[Dict[str, Any]] = None
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas")
    except Exception:  # older NumPy has no mode="dicts"
        pass
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "blas": {k: (blas or {}).get(k) for k in ("name", "version")},
        "pinned_env": {k: os.environ.get(k) for k in PINNED_ENV},
    }
