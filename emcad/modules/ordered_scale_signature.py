"""Ordered Scale Signature Lie-Flow neck.

The four backbone scales form an ordered path in a shared latent chart.  The
neck retains the first signature level (endpoint displacement) and the signed
second-level antisymmetric area, then applies the resulting truncated Euclidean
Lie action to each scale's local chart before the unchanged decoder348.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from emcad.registry import FLOW_REGISTRY


INNOVATION_NAME = "ordered_scale_lie_signature"
FLOW_NAME = "ordered_scale_lie_signature"
_SCALE_COUNT = 4
_EPS = 1.0e-6
_WRITE_LIMIT = 0.20
_SIGNATURE_RMS = 0.50
_TRANSLATION_LIMIT = 0.50
_UPSAMPLE_SCALES = (32, 16, 8, 4)


def _checked_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} must be finite")


def ordered_path_signature(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return level-1 endpoint displacement and level-2 signed area.

    For ordered points ``p_0,...,p_3`` with increments ``u_i``, the retained
    signature is

    ``S1 = sum_i u_i`` and
    ``A = 1/2 * sum_{i<j}(u_i⊗u_j - u_j⊗u_i)``.

    ``A`` is antisymmetric and changes sign under path orientation reversal or
    under an order swap that preserves endpoint displacement.  It therefore
    retains information absent from pooled means and unordered pairwise norms.
    """
    _checked_tensor(points, "points")
    if points.ndim != 3 or points.shape[1] != _SCALE_COUNT:
        raise ValueError("points must have shape [B,4,D]")
    increments = points[:, 1:] - points[:, :-1]
    first = increments.sum(dim=1)
    area = torch.zeros(
        points.shape[0], points.shape[-1], points.shape[-1],
        device=points.device, dtype=points.dtype,
    )
    for left in range(increments.shape[1]):
        for right in range(left + 1, increments.shape[1]):
            area = area + 0.5 * (
                torch.einsum("bi,bj->bij", increments[:, left], increments[:, right])
                - torch.einsum("bi,bj->bij", increments[:, right], increments[:, left])
            )
    return first, area


def _normalized_skew(area: torch.Tensor) -> torch.Tensor:
    skew = 0.5 * (area - area.transpose(-1, -2))
    rms = skew.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    return _SIGNATURE_RMS * skew / (rms + _EPS)


def lie_signature_matrix(
    first: torch.Tensor, area: torch.Tensor, time: float | torch.Tensor
) -> torch.Tensor:
    """Construct the orthogonal rotational part of the signature Lie action."""
    _checked_tensor(first, "first")
    _checked_tensor(area, "area")
    if first.ndim != 2 or area.ndim != 3 or area.shape[-1] != first.shape[-1]:
        raise ValueError("first and area have incompatible signature shapes")
    omega = _normalized_skew(area)
    if isinstance(time, torch.Tensor):
        time_value = time.to(device=omega.device, dtype=omega.dtype)
    else:
        time_value = torch.as_tensor(float(time), device=omega.device, dtype=omega.dtype)
    return torch.matrix_exp(omega * time_value)


def lie_signature_action(
    local: torch.Tensor, first: torch.Tensor, area: torch.Tensor, time: float | torch.Tensor
) -> torch.Tensor:
    """Apply a bounded SE(d)-style action to ``[B,D,N]`` local charts."""
    _checked_tensor(local, "local")
    if local.ndim != 3 or local.shape[0] != first.shape[0] or local.shape[1] != first.shape[1]:
        raise ValueError("local must have shape [B,D,N] matching first")
    matrix = lie_signature_matrix(first, area, time)
    if isinstance(time, torch.Tensor):
        time_value = time.to(device=local.device, dtype=local.dtype)
    else:
        time_value = torch.as_tensor(float(time), device=local.device, dtype=local.dtype)
    translation = _TRANSLATION_LIMIT * torch.tanh(first).unsqueeze(-1)
    return torch.bmm(matrix.to(local.dtype), local) + time_value * translation


def _power_of_two_scale(feature: torch.Tensor) -> torch.Tensor:
    maximum = feature.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
    safe = maximum.clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.floor(torch.log2(safe)).clamp(min=-126, max=127)
    return torch.where(maximum > 0.0, torch.exp2(exponent), torch.ones_like(exponent)).detach()


class OrderedScaleLieSignatureFlow(nn.Module):
    """Four-scale ordered-signature transport inserted at the EMCAD neck."""

    def __init__(self, channels: Sequence[int], latent_dim: int = 8) -> None:
        super().__init__()
        channels = [int(value) for value in channels]
        if len(channels) != _SCALE_COUNT or any(value <= 0 for value in channels):
            raise ValueError("OSSLF requires four positive channel counts")
        if not (2 <= int(latent_dim) <= 32):
            raise ValueError("latent_dim must be in [2,32]")
        self.channels = channels
        self.latent_dim = int(latent_dim)
        self.analysis = nn.ModuleList([nn.Linear(channel, self.latent_dim, bias=False) for channel in channels])
        self.synthesis = nn.ModuleList([nn.Linear(self.latent_dim, channel, bias=False) for channel in channels])
        self.signature_basis = nn.Parameter(torch.eye(self.latent_dim))
        self.action_gain = nn.Parameter(torch.zeros(()))
        self.latest_debug: dict[str, torch.Tensor] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.analysis:
            nn.init.orthogonal_(layer.weight)
        for layer in self.synthesis:
            nn.init.xavier_uniform_(layer.weight)

    def transform_features(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(features) != _SCALE_COUNT:
            raise ValueError("OSSLF requires four feature maps")
        latent_maps: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        observations: list[torch.Tensor] = []
        for index, (feature, channel, analysis) in enumerate(zip(features, self.channels, self.analysis)):
            _checked_tensor(feature, f"feature[{index}]")
            if feature.ndim != 4 or feature.shape[1] != channel:
                raise ValueError(f"feature[{index}] must have shape [B,{channel},H,W]")
            work = feature.float()
            scale = _power_of_two_scale(work)
            normalized = work / scale
            latent = torch.einsum("dc,bchw->bdhw", analysis.weight.float(), normalized)
            latent_maps.append(latent)
            scales.append(scale)
            observations.append(latent.mean(dim=(-2, -1)))
        if len({value.shape[0] for value in features}) != 1:
            raise ValueError("all OSSLF feature maps must share the batch dimension")

        points = torch.stack(observations, dim=1)
        first, raw_area = ordered_path_signature(points)
        basis = self.signature_basis.float()
        transformed_area = torch.bmm(torch.bmm(basis.unsqueeze(0).expand(points.shape[0], -1, -1), raw_area), basis.transpose(0, 1).unsqueeze(0).expand(points.shape[0], -1, -1))
        gain = torch.tanh(self.action_gain.float())
        outputs: list[torch.Tensor] = []
        displacement_maps: list[torch.Tensor] = []
        times = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
        for index, (feature, latent, scale, synthesis, time) in enumerate(
            zip(features, latent_maps, scales, self.synthesis, times)
        ):
            centered = latent - points[:, index, :, None, None]
            flat = centered.reshape(centered.shape[0], self.latent_dim, -1)
            transported = lie_signature_action(flat, first, transformed_area, time)
            delta = (transported - flat).reshape_as(centered)
            decoded = torch.einsum("cd,bdhw->bchw", synthesis.weight.float(), delta)
            bounded = _WRITE_LIMIT * scale * torch.tanh(decoded)
            actual = gain * bounded
            outputs.append(feature + actual.to(feature.dtype))
            displacement_maps.append(actual.detach().abs().mean(dim=(1, 2, 3), keepdim=True))

        self.latest_debug = {
            "signature_first_level": first.detach(),
            "signature_raw_area": raw_area.detach(),
            "signature_area": transformed_area.detach(),
            "signature_action_gain": gain.detach().reshape(1, 1, 1, 1),
            "signature_write_magnitude": torch.cat(displacement_maps, dim=1),
        }
        return tuple(outputs)

    def encode(self, model, x):
        x = model.preprocess_input(x)
        return self.transform_features(tuple(model.backbone(x)))

    def decode(self, model, encoded):
        x1, x2, x3, x4 = encoded
        return model.decoder(x4, [x3, x2, x1])

    def predict(self, model, decoded):
        return [head(feature) for head, feature in zip(model.get_prediction_heads(), decoded)]

    def upsample(self, model, predictions):
        del model
        return [
            F.interpolate(prediction, scale_factor=scale, mode="bilinear", align_corners=False)
            for prediction, scale in zip(predictions, _UPSAMPLE_SCALES)
        ]


@FLOW_REGISTRY.register(FLOW_NAME)
def build_ordered_scale_lie_signature_flow(*, config: dict[str, Any]):
    if not isinstance(config, dict):
        raise TypeError("OSSLF config must be a mapping")
    unknown = set(config) - {"name", "channels", "latent_dim"}
    if unknown:
        raise ValueError(f"unknown OSSLF config keys: {sorted(unknown)}")
    channels = config.get("channels")
    if not isinstance(channels, list):
        raise ValueError("OSSLF config requires channels")
    return OrderedScaleLieSignatureFlow(channels, int(config.get("latent_dim", 8)))


__all__ = [
    "FLOW_NAME",
    "INNOVATION_NAME",
    "OrderedScaleLieSignatureFlow",
    "build_ordered_scale_lie_signature_flow",
    "lie_signature_action",
    "lie_signature_matrix",
    "ordered_path_signature",
]
