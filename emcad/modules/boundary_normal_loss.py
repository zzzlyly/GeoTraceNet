"""Boundary-normal alignment training objective."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from emcad.registry import LOSS_REGISTRY

LOSS_NAME = "structure_loss_radialrankboundary_boundary_normal_alignment"
BASE_LOSS_NAME = "structure_loss_radialrankboundary"
NORMAL_WEIGHT = 0.08
NORMAL_EPS = 1.0e-4


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    y = x.transpose(0, 1).contiguous()
    return x.reshape(1, 1, 3, 3), y.reshape(1, 1, 3, 3)


def _morphological_boundary(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_normal_alignment_penalty(
    pred: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = NORMAL_EPS,
) -> torch.Tensor:
    """Align predicted probability gradients with mask boundary gradients."""
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")
    target = mask.float().clamp(0.0, 1.0)
    probability = torch.sigmoid(pred.float())
    kernel_x, kernel_y = _sobel_kernels(probability.device, probability.dtype)
    pred_x = F.conv2d(probability, kernel_x, padding=1)
    pred_y = F.conv2d(probability, kernel_y, padding=1)
    target_x = F.conv2d(target, kernel_x, padding=1)
    target_y = F.conv2d(target, kernel_y, padding=1)
    pred_norm = (pred_x.square() + pred_y.square() + float(eps) ** 2).sqrt()
    target_norm = (target_x.square() + target_y.square() + float(eps) ** 2).sqrt()
    cosine = (pred_x * target_x + pred_y * target_y) / (pred_norm * target_norm)
    boundary = _morphological_boundary(target)
    denominator = boundary.sum(dim=(-2, -1)).clamp_min(float(eps))
    return ((1.0 - cosine) * boundary).sum(dim=(-2, -1)).div(denominator).mean()


@LOSS_REGISTRY.register(LOSS_NAME)
def structure_loss_radialrankboundary_boundary_normal_alignment(
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
    eps: float = 1.0e-6,
    normal_weight: float = NORMAL_WEIGHT,
    normal_eps: float = NORMAL_EPS,
) -> torch.Tensor:
    """Radial-rank-boundary objective plus boundary-normal alignment."""
    base = LOSS_REGISTRY.get(BASE_LOSS_NAME)(
        pred, mask, weight=weight, base_weight=base_weight,
        rank_weight=rank_weight, halo_weight=halo_weight,
        core_radius=core_radius, boundary_radius=boundary_radius,
        inner_halo_radius=inner_halo_radius, outer_halo_radius=outer_halo_radius,
        margin=margin, eps=eps,
    )
    normal = boundary_normal_alignment_penalty(pred, mask, eps=normal_eps)
    return base + float(weight) * float(normal_weight) * normal
