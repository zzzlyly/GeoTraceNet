"""Boundary-valid two-tail-trimmed anchored local barycenter head.

The head reads one Golden--Thompson certificate ratio, removes one local
minimum and one local maximum from each real-position 3x3 multiset, and
averages the retained location with the centre anchor.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from emcad.registry import HEAD_REGISTRY
from emcad.modules.trimmed_common import (
    BASE_HEAD,
    CERTIFICATE,
    NEIGHBOURS,
    assert_finite_bounded,
    build_heads,
    logits_from_ratio,
    shift_replicate,
)

HEAD_NAME = "trimmed_barycenter_4scale"
STENCIL = ((0, 0),) + NEIGHBOURS


def _work_dtype(value: torch.Tensor) -> torch.dtype:
    if value.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return value.dtype


def _valid_neighbour(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = value.shape[-2:]
    y = torch.arange(height, device=value.device).reshape(1, 1, height, 1)
    x = torch.arange(width, device=value.device).reshape(1, 1, 1, width)
    return (y + dy >= 0) & (y + dy < height) & (x + dx >= 0) & (x + dx < width)


def boundary_valid_two_tail_trimmed_barycenter(
    ratio: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    input_dtype = ratio.dtype
    ratio = ratio.to(dtype=_work_dtype(ratio)).clamp(0.0, 1.0)
    residuals: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for dy, dx in STENCIL:
        shifted = ratio if (dy, dx) == (0, 0) else shift_replicate(ratio, dy, dx)
        valid = _valid_neighbour(ratio, dy, dx)
        residuals.append(shifted - ratio)
        valids.append(valid)

    residual_stack = torch.stack(residuals, dim=0)
    valid_stack = torch.stack(valids, dim=0)
    zero = torch.zeros_like(residual_stack)
    residual_sum = torch.where(valid_stack, residual_stack, zero).sum(dim=0)
    residual_min = torch.where(
        valid_stack, residual_stack, torch.full_like(residual_stack, 2.0)
    ).amin(dim=0)
    residual_max = torch.where(
        valid_stack, residual_stack, torch.full_like(residual_stack, -2.0)
    ).amax(dim=0)
    count = valid_stack.sum(dim=0).to(dtype=ratio.dtype).expand_as(ratio)
    eligible = count >= 3.0
    safe_denominator = torch.where(eligible, count - 2.0, torch.ones_like(count))
    candidate_residual = (
        residual_sum - residual_min - residual_max
    ) / safe_denominator
    trimmed_residual = torch.where(
        eligible, candidate_residual, torch.zeros_like(candidate_residual)
    )
    output = ratio + 0.5 * trimmed_residual
    return output.to(dtype=input_dtype), {
        "count": count,
        "residual_sum": residual_sum,
        "residual_min": residual_min,
        "residual_max": residual_max,
        "trimmed_residual": trimmed_residual,
    }


class BoundaryValidTwoTailTrimmedBarycenterHead(BASE_HEAD):
    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__(in_channels, num_classes=num_classes)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        dtype = feature.dtype
        first, second = self.semantic_hamiltonians(feature)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            certificate = CERTIFICATE(first, second)
            ratio = certificate["golden_tightness_ratio"]
            improved, witnesses = boundary_valid_two_tail_trimmed_barycenter(ratio)
            logits = logits_from_ratio(self, improved, dtype)
        assert_finite_bounded(improved, "trimmed_barycenter")
        self.latest_maps = {
            **certificate,
            "trimmed_ratio": improved.detach(),
            **{
                f"trimmed_{name}": value.detach()
                for name, value in witnesses.items()
            },
        }
        self.latest_debug = {
            "trimmed_valid_count_mean": witnesses["count"]
            .detach()
            .mean(dim=(-2, -1)),
            "trimmed_residual_abs_mean": witnesses["trimmed_residual"]
            .detach()
            .abs()
            .mean(dim=(-2, -1)),
        }
        return logits


@HEAD_REGISTRY.register(HEAD_NAME)
def build_trimmed_barycenter_heads(*, channels: list[int], config: dict) -> nn.ModuleList:
    return build_heads(
        BoundaryValidTwoTailTrimmedBarycenterHead,
        channels=channels,
        config=config,
        name=HEAD_NAME,
    )
