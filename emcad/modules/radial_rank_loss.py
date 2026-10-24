from __future__ import annotations

import torch
import torch.nn.functional as F

from emcad.registry import LOSS_REGISTRY


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=radius)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _region_mean(values: torch.Tensor, region: torch.Tensor, eps: float) -> torch.Tensor:
    denom = region.sum(dim=(2, 3)).clamp_min(eps)
    return (values * region).sum(dim=(2, 3)) / denom


def _structure_loss(pred: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    kernel = min(31, max(3, ((min(mask.shape[-2:]) // 2) * 2) - 1))
    if kernel % 2 == 0:
        kernel -= 1
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=kernel, stride=1, padding=kernel // 2) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(eps)

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def _radial_regions(
    mask: torch.Tensor,
    core_radius: int,
    boundary_radius: int,
    inner_halo_radius: int,
    outer_halo_radius: int,
) -> dict[str, torch.Tensor]:
    mask = mask.float().clamp(0.0, 1.0)
    core = _erode(mask, core_radius)
    inner_edge = _erode(mask, boundary_radius)
    outer_edge = _dilate(mask, boundary_radius)
    boundary = (outer_edge - inner_edge).clamp(0.0, 1.0)
    inner_halo = (_dilate(mask, inner_halo_radius) - mask).clamp(0.0, 1.0)
    outer_halo = (_dilate(mask, outer_halo_radius) - _dilate(mask, inner_halo_radius)).clamp(0.0, 1.0)
    far_background = (1.0 - _dilate(mask, outer_halo_radius)).clamp(0.0, 1.0)
    return {
        "core": core,
        "boundary": boundary,
        "inner_halo": inner_halo,
        "outer_halo": outer_halo,
        "far_background": far_background,
    }


def _radial_rank_loss(
    prob: torch.Tensor,
    mask: torch.Tensor,
    core_radius: int,
    boundary_radius: int,
    inner_halo_radius: int,
    outer_halo_radius: int,
    margin: float,
    eps: float,
) -> torch.Tensor:
    regions = _radial_regions(mask, core_radius, boundary_radius, inner_halo_radius, outer_halo_radius)
    core = _region_mean(prob, regions["core"], eps)
    boundary = _region_mean(prob, regions["boundary"], eps)
    inner_halo = _region_mean(prob, regions["inner_halo"], eps)
    outer_halo = _region_mean(prob, regions["outer_halo"], eps)
    far_background = _region_mean(prob, regions["far_background"], eps)
    lesion_gate = (mask.flatten(1).sum(dim=1, keepdim=True) > eps).float()

    rank = (
        F.relu(margin - (core - boundary))
        + F.relu(margin - (boundary - inner_halo))
        + F.relu(0.5 * margin - (inner_halo - outer_halo))
        + F.relu(0.5 * margin - (outer_halo - far_background))
    )
    halo_excess = _region_mean(F.relu(prob - 0.20).square(), regions["inner_halo"], eps)
    far_excess = _region_mean(F.relu(prob - 0.08).square(), regions["far_background"], eps)
    return (lesion_gate * rank + halo_excess + far_excess).mean()


@LOSS_REGISTRY.register("structure_loss_radialrankboundary")
def structure_loss_radialrankboundary(
    pred: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 1.0,
    base_weight: float = 1.0,
    rank_weight: float = 0.18,
    halo_weight: float = 0.08,
    core_radius: int = 2,
    boundary_radius: int = 1,
    inner_halo_radius: int = 4,
    outer_halo_radius: int = 8,
    margin: float = 0.08,
    eps: float = 1e-6,
) -> torch.Tensor:
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")
    mask = mask.float()
    prob = torch.sigmoid(pred)
    regions = _radial_regions(mask, core_radius, boundary_radius, inner_halo_radius, outer_halo_radius)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    boundary_weight_map = 1.0 + 2.0 * regions["boundary"] + 0.5 * regions["inner_halo"]
    band_bce = (bce * boundary_weight_map).sum(dim=(2, 3)) / boundary_weight_map.sum(dim=(2, 3)).clamp_min(eps)
    rank = _radial_rank_loss(
        prob,
        mask,
        core_radius,
        boundary_radius,
        inner_halo_radius,
        outer_halo_radius,
        margin,
        eps,
    )
    halo = _region_mean(F.relu(prob - 0.15).square(), regions["inner_halo"] + regions["outer_halo"], eps).mean()
    return weight * (base_weight * _structure_loss(pred, mask, eps) + band_bce.mean() + rank_weight * rank + halo_weight * halo)

