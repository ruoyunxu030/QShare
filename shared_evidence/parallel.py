"""CPU parallelism: one BLAS thread per process; worker count defaults to CPU cores minus 4."""

from __future__ import annotations

import os
from dataclasses import dataclass


BLAS_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_cpu_threads(n: int = 1) -> None:
    """Must run before numpy/sklearn import in a fresh interpreter."""
    s = str(int(n))
    for k in BLAS_KEYS:
        os.environ.setdefault(k, s)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(n))
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(int(n))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


@dataclass(frozen=True)
class ParallelPlan:
    device: str
    n_workers: int
    threads_per_worker: int
    n_cpu: int
    n_gpu: int
    reason: str


def _cpu_count() -> int:
    return int(os.cpu_count() or 8)


def _gpu_count() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        return 0
    return 0


def choose_parallelism(kind: str = "market") -> ParallelPlan:
    """CPU market jobs: 1 BLAS thread per process. QSHARE_WORKERS overrides the count."""
    n_cpu = _cpu_count()
    n_gpu = _gpu_count()
    env = os.environ.get("QSHARE_WORKERS", "").strip()
    default = max(1, min(62, n_cpu - 4))
    nw = int(env) if env else default
    return ParallelPlan(
        device="cpu", n_workers=max(1, nw), threads_per_worker=1,
        n_cpu=n_cpu, n_gpu=n_gpu,
        reason=f"1 BLAS thread × {nw} processes (default n_cpu-4).",
    )
