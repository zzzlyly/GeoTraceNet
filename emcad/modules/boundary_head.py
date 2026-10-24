"""Certificate head used with the boundary-normal training objective."""
from __future__ import annotations

import torch
import torch.nn as nn

from emcad.registry import HEAD_REGISTRY
from emcad.modules.boundary_common import (
    BASE_HEAD,
    build_heads,
    certificate_ratio,
    finish,
    logits_from_coordinate,
)

HEAD_NAME = "boundary_valid_two_tail_trimmed_barycenter_4scale"
USE_CERTIFICATE_DIAGNOSTICS = False


class BoundaryValidHead(BASE_HEAD):
    """Certificate head for the boundary-normal training objective."""

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        dtype = feature.dtype
        ratio, maps = certificate_ratio(self, feature)
        coordinate = (2.0 * ratio - 1.0).to(dtype=dtype)
        logits = logits_from_coordinate(self, coordinate, dtype)
        maps.update(
            {
                "boundary_ratio": ratio,
                "boundary_coordinate": coordinate,
            }
        )
        return finish(self, maps, coordinate, logits, "boundary")


@HEAD_REGISTRY.register(HEAD_NAME)
def build_boundary_heads(
    *, channels: list[int], config: dict
) -> nn.ModuleList:
    return build_heads(
        BoundaryValidHead,
        channels=channels,
        config=config,
        name=HEAD_NAME,
    )
