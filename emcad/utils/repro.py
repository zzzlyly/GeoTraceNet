from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_seed_list(seed_values: str | Iterable[int] | None, default_seed: int) -> list[int]:
    if seed_values is None:
        return [default_seed]
    if isinstance(seed_values, (list, tuple)):
        values = [int(item) for item in seed_values]
        return values if values else [default_seed]
    if isinstance(seed_values, str):
        items = [item.strip() for item in seed_values.split(",") if item.strip()]
        return [int(item) for item in items] if items else [default_seed]
    return [int(seed_values)]


def resolve_run_seeds(
    seed_values: str | Iterable[int] | None,
    default_seed: int,
    runs: int | None,
) -> list[int]:
    run_count = int(runs or 1)
    if run_count <= 0:
        raise ValueError(f"runs must be >= 1, got {run_count}")

    if seed_values is None:
        return [default_seed + offset for offset in range(run_count)]
    if isinstance(seed_values, str) and not seed_values.strip():
        return [default_seed + offset for offset in range(run_count)]
    if isinstance(seed_values, (list, tuple)) and len(seed_values) == 0:
        return [default_seed + offset for offset in range(run_count)]
    return parse_seed_list(seed_values, default_seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_reproducibility(seed: int, deterministic: bool, *, strict: bool = False) -> None:
    """Configure process-level reproducibility controls."""
    if strict and not deterministic:
        raise ValueError("strict reproducibility requires deterministic=true")
    # cuBLAS reads this at CUDA-context creation, so set it before seeding.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if strict and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "deterministic_v3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization"
        )
    set_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Enforce deterministic kernels when strict mode is enabled.
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
