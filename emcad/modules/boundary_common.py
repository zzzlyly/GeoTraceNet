from __future__ import annotations

import importlib
import torch
import torch.nn.functional as F

from emcad.modules.trimmed_common import (
    assert_finite_bounded,
    build_heads,
    shift_replicate,
)

BASE = importlib.import_module("emcad_ext_gibbs_compatibility_head")
TRIMMED = importlib.import_module("emcad_ext_trimmed_barycenter")
BASE_HEAD = BASE.GoldenThompsonGibbsCompatibilityHead
CERTIFICATE = BASE.golden_thompson_gibbs_certificate
BOUNDARY = TRIMMED.boundary_valid_two_tail_trimmed_barycenter
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
FLOOR = 0.025
EPS = 1.0e-8


def certificate_ratio(head: torch.nn.Module, feature: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.autocast(device_type=feature.device.type, enabled=False):
        first, second = head.semantic_hamiltonians(feature)
        certificate = CERTIFICATE(first, second, diagnostics=False)
        ratio, witnesses = BOUNDARY(certificate["golden_tightness_ratio"])
    return ratio, {**certificate, **{f"trimmed_{key}": value for key, value in witnesses.items()}}


def global_standardized(value: torch.Tensor, floor: float = FLOOR) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work = value.float()
    mean = work.mean(dim=(-2, -1), keepdim=True)
    centred = work - mean
    std = (centred.square().mean(dim=(-2, -1), keepdim=True) + float(floor) ** 2).sqrt()
    return centred / std, mean, std


def logits_from_coordinate(head: torch.nn.Module, coordinate: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    logits = head.gibbs_scale.float() * coordinate.float() + head.gibbs_bias.float().reshape(1, 1, 1, 1)
    return logits.to(dtype)


def finish(head: torch.nn.Module, maps: dict[str, torch.Tensor], coordinate: torch.Tensor, logits: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.isfinite(coordinate).all():
        raise FloatingPointError(f"{name} produced non-finite coordinates")
    if coordinate.min().item() < -1.0004 or coordinate.max().item() > 1.0004:
        raise FloatingPointError(f"{name} left the signed coordinate interval")
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"{name} produced non-finite logits")
    head.latest_maps = {key: value.detach() for key, value in maps.items()}
    head.latest_maps[f"{name}_coordinate"] = coordinate.detach()
    head.latest_maps[f"{name}_logit"] = logits.detach()
    head.latest_debug = {
        f"{name}_coordinate_mean": coordinate.detach().mean(dim=(-2, -1)),
        f"{name}_coordinate_std": coordinate.detach().std(dim=(-2, -1), unbiased=False),
    }
    return logits


def local_stack(value: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [value] + [shift_replicate(value, dy, dx) for dy, dx in NEIGHBOURS], dim=0
    )


def local_mean_std(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stack = local_stack(value)
    mean = stack.mean(dim=0)
    std = (stack - mean).square().mean(dim=0).add(FLOOR ** 2).sqrt()
    return mean, std
