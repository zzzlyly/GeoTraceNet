"""Shared scalar operators for the local certificate heads."""
from __future__ import annotations

import torch
from emcad.modules.certificate_common import (
    BASE_HEAD,
    CERTIFICATE,
    NEIGHBOURS,
    assert_finite_bounded,
    build_heads,
    shift_replicate,
)

EPS = 1.0e-8


def logits_from_ratio(head: torch.nn.Module, ratio: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    coordinate = 2.0 * ratio - 1.0
    logits = head.gibbs_scale.float() * coordinate + head.gibbs_bias.float().reshape(1, 1, 1, 1)
    return logits.to(dtype)


__all__ = [
    "BASE_HEAD",
    "CERTIFICATE",
    "NEIGHBOURS",
    "assert_finite_bounded",
    "build_heads",
    "logits_from_ratio",
    "shift_replicate",
]
