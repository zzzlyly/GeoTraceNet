"""Shared certificate and compact scalar-map operators."""
from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F

BASE = importlib.import_module("emcad_ext_gibbs_compatibility_head")
BASE_HEAD = BASE.GoldenThompsonGibbsCompatibilityHead
CERTIFICATE = BASE.golden_thompson_gibbs_certificate

EPS = 1.0e-8
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def shift_replicate(value: torch.Tensor, dy: int, dx: int, radius: int = 1) -> torch.Tensor:
    height, width = value.shape[-2:]
    pad = int(radius)
    padded = F.pad(value, (pad, pad, pad, pad), mode="replicate")
    return padded[:, :, pad + dy * radius : pad + dy * radius + height, pad + dx * radius : pad + dx * radius + width]


def assert_finite_bounded(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} produced non-finite values")
    if value.min().item() < -1.0e-4 or value.max().item() > 1.0004:
        raise FloatingPointError(f"{name} left the certificate interval")


def build_heads(head_cls, *, channels: list[int], config: dict, name: str):
    import torch.nn as nn
    if not isinstance(channels, list) or len(channels) != 4:
        raise ValueError(f"{name} requires four scales")
    if int(config.get("num_classes", 1)) != 1 or set(config) - {"num_classes"}:
        raise ValueError(f"{name} accepts only num_classes")
    return nn.ModuleList([head_cls(int(channel), num_classes=1) for channel in channels])
