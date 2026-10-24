r"""Golden--Thompson Gibbs-compatibility prediction head.

Decoder features are mapped to two real symmetric Hamiltonian fields. The head
compares their joint and split Gibbs partitions, then converts the bounded
compatibility coordinate into a segmentation logit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from emcad.registry import HEAD_REGISTRY


_EPS = 1.0e-8
_NORMALIZE_EPS = 1.0e-6
_ORDER = 4
_SYMMETRIC_FIELDS = _ORDER * (_ORDER + 1) // 2
_TOTAL_FIELDS = 2 * _SYMMETRIC_FIELDS
_FIELD_BOUND = 0.9
INNOVATION_NAME = "gibbs_compatibility_head"
HEAD_NAME = "golden_thompson_gibbs_compatibility_4scale"


def symmetric_matrix_from_fields(fields: torch.Tensor) -> torch.Tensor:
    """Expand ten labelled fields to ``[B,4,4,H,W]`` symmetric matrices."""
    if fields.ndim != 4 or fields.shape[1] != _SYMMETRIC_FIELDS:
        raise ValueError("fields must have shape [B, 10, H, W]")
    if min(fields.shape[-2:]) < 1:
        raise ValueError("symmetric fields require non-empty spatial dimensions")
    if not fields.is_floating_point():
        raise TypeError("symmetric fields must be floating point")

    pairs = (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )
    basis = fields.new_zeros((_SYMMETRIC_FIELDS, _ORDER, _ORDER))
    for index, (row, column) in enumerate(pairs):
        basis[index, row, column] = 1.0
        basis[index, column, row] = 1.0
    return torch.einsum("bkhw,kij->bijhw", fields, basis)


def trace_center_hilbert_schmidt(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace-centre a symmetric matrix and normalize ``tau(H^2)`` toward one."""
    if matrix.ndim != 5 or matrix.shape[1:3] != (_ORDER, _ORDER):
        raise ValueError("matrix must have shape [B, 4, 4, H, W]")
    if not matrix.is_floating_point():
        raise TypeError("matrix must be floating point")
    work = matrix.to(torch.float64 if matrix.dtype == torch.float64 else torch.float32)
    if not torch.isfinite(work).all():
        raise FloatingPointError("Hamiltonian fields contain non-finite values")
    symmetry_error = (work - work.transpose(1, 2)).abs().amax(dim=(1, 2), keepdim=True)
    tolerance = 2.0e-10 if work.dtype == torch.float64 else 2.0e-5
    if symmetry_error.max().item() > tolerance:
        raise ValueError("Hamiltonian matrix must be symmetric")

    diagonal = torch.stack(
        [work[:, index, index] for index in range(_ORDER)], dim=1
    )
    mean_trace = diagonal.mean(dim=1, keepdim=True)
    eye = torch.eye(_ORDER, device=work.device, dtype=work.dtype).reshape(
        1, _ORDER, _ORDER, 1, 1
    )
    centred = work - mean_trace.unsqueeze(2) * eye
    energy = centred.square().sum(dim=(1, 2), keepdim=True) / float(_ORDER)
    normalizer = (energy + _NORMALIZE_EPS).sqrt()
    standardized = centred / normalizer
    return standardized, mean_trace, normalizer.squeeze(2)


def _to_trailing(matrix: torch.Tensor) -> torch.Tensor:
    # ``torch.matrix_exp`` requires a reshape-compatible batched matrix view;
    # materialize the spatial-major layout, including constant/expanded fixtures.
    return matrix.permute(0, 3, 4, 1, 2).contiguous()


def _from_trailing(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.permute(0, 3, 4, 1, 2)


def _normalized_trace(matrix: torch.Tensor) -> torch.Tensor:
    """Normalized trace of ``[B,H,W,4,4]`` matrices as ``[B,1,H,W]``."""
    return matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(1) / float(
        _ORDER
    )


def golden_thompson_gibbs_certificate(
    first_hamiltonian: torch.Tensor,
    second_hamiltonian: torch.Tensor,
    *,
    diagnostics: bool = True,
) -> dict[str, torch.Tensor]:
    """Evaluate the Golden--Thompson joint/split Gibbs partition certificate."""
    if first_hamiltonian.ndim != 5 or first_hamiltonian.shape[1:3] != (
        _ORDER,
        _ORDER,
    ):
        raise ValueError("first_hamiltonian must have shape [B, 4, 4, H, W]")
    if second_hamiltonian.shape != first_hamiltonian.shape:
        raise ValueError("second_hamiltonian must share shape [B, 4, 4, H, W]")
    if min(first_hamiltonian.shape[-2:]) < 1:
        raise ValueError("Gibbs fields require non-empty spatial dimensions")
    if not first_hamiltonian.is_floating_point() or not second_hamiltonian.is_floating_point():
        raise TypeError("Gibbs Hamiltonians must be floating point")

    work_dtype = (
        torch.float64
        if first_hamiltonian.dtype == torch.float64
        and second_hamiltonian.dtype == torch.float64
        else torch.float32
    )
    first = first_hamiltonian.to(work_dtype)
    second = second_hamiltonian.to(work_dtype)
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError("Gibbs Hamiltonians contain non-finite values")
    first_symmetry_error = (
        first - first.transpose(1, 2)
    ).abs().amax(dim=(1, 2), keepdim=True)
    second_symmetry_error = (
        second - second.transpose(1, 2)
    ).abs().amax(dim=(1, 2), keepdim=True)
    symmetry_tolerance = 2.0e-10 if work_dtype == torch.float64 else 2.0e-5
    if first_symmetry_error.max().item() > symmetry_tolerance or second_symmetry_error.max().item() > symmetry_tolerance:
        raise ValueError("Gibbs Hamiltonians must be symmetric")

    first_trailing = _to_trailing(first)
    second_trailing = _to_trailing(second)
    joint_trailing = first_trailing + second_trailing
    exp_first = torch.matrix_exp(first_trailing)
    exp_second = torch.matrix_exp(second_trailing)
    exp_joint = torch.matrix_exp(joint_trailing)
    split_product = exp_first @ exp_second
    split_partition = _normalized_trace(split_product)
    joint_partition = _normalized_trace(exp_joint)
    if torch.any(split_partition <= 0.0) or torch.any(joint_partition <= 0.0):
        raise FloatingPointError("Gibbs partition lost strict positivity")

    tightness_ratio = joint_partition / split_partition
    golden_gap = split_partition.log() - joint_partition.log()
    coordinate = 2.0 * tightness_ratio - 1.0
    inequality_slack = split_partition - joint_partition

    if not diagnostics:
        maps = {
            "golden_split_partition": split_partition,
            "golden_joint_partition": joint_partition,
            "golden_tightness_ratio": tightness_ratio,
            "golden_gap": golden_gap,
            "golden_coordinate": coordinate,
            "golden_inequality_slack": inequality_slack,
        }
        if not all(torch.isfinite(value).all() for value in maps.values()):
            raise FloatingPointError("non-finite Golden--Thompson Gibbs certificate")
        theorem_tolerance = 3.0e-9 if work_dtype == torch.float64 else 3.0e-4
        if torch.any(tightness_ratio <= 0.0):
            raise FloatingPointError("Golden--Thompson ratio lost positivity")
        if torch.any(tightness_ratio > 1.0 + theorem_tolerance):
            raise FloatingPointError("Golden--Thompson inequality was violated")
        if torch.any(coordinate < -1.0 - theorem_tolerance) or torch.any(
            coordinate > 1.0 + 2.0 * theorem_tolerance
        ):
            raise FloatingPointError("Golden--Thompson coordinate escaped [-1, 1]")
        return maps

    product_ab = first_trailing @ second_trailing
    first_squared = first_trailing @ first_trailing
    second_squared = second_trailing @ second_trailing
    product_abab = product_ab @ product_ab
    product_a2b2 = first_squared @ second_squared
    commutator = product_ab - second_trailing @ first_trailing
    commutator_energy = commutator.square().sum(dim=(-2, -1)).unsqueeze(1) / float(
        _ORDER
    )
    trace_aa = _normalized_trace(first_squared)
    trace_ab = _normalized_trace(product_ab)
    trace_bb = _normalized_trace(second_squared)
    trace_gram = torch.stack(
        (
            torch.stack((trace_aa[:, 0], trace_ab[:, 0]), dim=1),
            torch.stack((trace_ab[:, 0], trace_bb[:, 0]), dim=1),
        ),
        dim=1,
    )
    trace_abab = _normalized_trace(product_abab)
    trace_a2b2 = _normalized_trace(product_a2b2)
    alternating_free_cumulant = trace_abab - 2.0 * trace_ab.square()
    
    # Compute eigenvalues in chunks to avoid OOM from CUDA eigvalsh workspace
    # first_trailing/second_trailing shape: [B, H, W, 4, 4]
    B, H, W = first_trailing.shape[:3]
    chunk_size = 8  # Process 8 batches at a time to limit peak memory
    first_eigenvalues_list = []
    second_eigenvalues_list = []
    
    for i in range(0, B, chunk_size):
        end_i = min(i + chunk_size, B)
        first_chunk = first_trailing[i:end_i]
        second_chunk = second_trailing[i:end_i]
        first_eig_chunk = torch.linalg.eigvalsh(first_chunk)  # [chunk, H, W, 4]
        second_eig_chunk = torch.linalg.eigvalsh(second_chunk)
        first_eigenvalues_list.append(first_eig_chunk)
        second_eigenvalues_list.append(second_eig_chunk)
    
    first_eigenvalues = torch.cat(first_eigenvalues_list, dim=0).permute(0, 3, 1, 2)
    second_eigenvalues = torch.cat(second_eigenvalues_list, dim=0).permute(0, 3, 1, 2)
    first_trace = _normalized_trace(first_trailing)
    second_trace = _normalized_trace(second_trailing)

    maps = {
        "golden_first_hamiltonian": first,
        "golden_second_hamiltonian": second,
        "golden_exp_first": _from_trailing(exp_first),
        "golden_exp_second": _from_trailing(exp_second),
        "golden_exp_joint": _from_trailing(exp_joint),
        "golden_split_product": _from_trailing(split_product),
        "golden_split_partition": split_partition,
        "golden_joint_partition": joint_partition,
        "golden_tightness_ratio": tightness_ratio,
        "golden_gap": golden_gap,
        "golden_coordinate": coordinate,
        "golden_inequality_slack": inequality_slack,
        "golden_commutator": _from_trailing(commutator),
        "golden_commutator_energy": commutator_energy,
        "golden_trace_gram": trace_gram,
        "golden_trace_abab": trace_abab,
        "golden_trace_a2b2": trace_a2b2,
        "golden_alternating_free_cumulant": alternating_free_cumulant,
        "golden_first_eigenvalues": first_eigenvalues,
        "golden_second_eigenvalues": second_eigenvalues,
        "golden_first_trace": first_trace,
        "golden_second_trace": second_trace,
        "golden_first_symmetry_error": first_symmetry_error.squeeze(2),
        "golden_second_symmetry_error": second_symmetry_error.squeeze(2),
    }
    if not all(torch.isfinite(value).all() for value in maps.values()):
        raise FloatingPointError("non-finite Golden--Thompson Gibbs certificate")
    theorem_tolerance = 3.0e-9 if work_dtype == torch.float64 else 3.0e-4
    if torch.any(tightness_ratio <= 0.0):
        raise FloatingPointError("Golden--Thompson ratio lost positivity")
    if torch.any(tightness_ratio > 1.0 + theorem_tolerance):
        raise FloatingPointError("Golden--Thompson inequality was violated")
    if torch.any(coordinate < -1.0 - theorem_tolerance) or torch.any(
        coordinate > 1.0 + 2.0 * theorem_tolerance
    ):
        raise FloatingPointError("Golden--Thompson coordinate escaped [-1, 1]")
    return maps


class GoldenThompsonGibbsCompatibilityHead(nn.Module):
    """One native-scale predictor from an all-orders Gibbs partition ratio."""

    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if num_classes != 1:
            raise ValueError("Golden--Thompson head supports num_classes == 1")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.mechanism_projection = nn.Conv2d(
            self.in_channels, _TOTAL_FIELDS, kernel_size=1
        )
        self.log_gibbs_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32)
        )
        self.gibbs_bias = nn.Parameter(torch.zeros(1))
        self.latest_debug: dict[str, torch.Tensor] = {}
        self.latest_maps: dict[str, torch.Tensor] = {}

    @property
    def gibbs_scale(self) -> torch.Tensor:
        return F.softplus(self.log_gibbs_scale) + _EPS

    def semantic_hamiltonians(
        self, feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError("head feature must have shape [B, C, H, W]")
        if feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} feature channels, got {feature.shape[1]}"
            )
        fields = _FIELD_BOUND * torch.tanh(self.mechanism_projection(feature).float())
        first_matrix = symmetric_matrix_from_fields(fields[:, :_SYMMETRIC_FIELDS])
        second_matrix = symmetric_matrix_from_fields(fields[:, _SYMMETRIC_FIELDS:])
        first, _, _ = trace_center_hilbert_schmidt(first_matrix)
        second, _, _ = trace_center_hilbert_schmidt(second_matrix)
        return first, second

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        output_dtype = feature.dtype
        first, second = self.semantic_hamiltonians(feature)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            certificate = golden_thompson_gibbs_certificate(first, second)
            coordinate = certificate["golden_coordinate"]
            logits = (
                self.gibbs_scale.float() * coordinate
                + self.gibbs_bias.float().reshape(1, 1, 1, 1)
            )
        self.latest_debug = {
            "golden_coordinate_mean": coordinate.detach().mean(dim=(-2, -1)),
            "golden_ratio_mean": certificate["golden_tightness_ratio"]
            .detach()
            .mean(dim=(-2, -1)),
            "golden_gap_mean": certificate["golden_gap"]
            .detach()
            .mean(dim=(-2, -1)),
            "golden_slack_mean": certificate["golden_inequality_slack"]
            .detach()
            .mean(dim=(-2, -1)),
            "golden_commutator_energy_mean": certificate[
                "golden_commutator_energy"
            ].detach().mean(dim=(-2, -1)),
            "golden_gibbs_scale": self.gibbs_scale.detach()
            .reshape(1, 1)
            .expand(feature.shape[0], -1),
        }
        self.latest_maps = {
            name: value.detach() for name, value in certificate.items()
        }
        return logits.to(output_dtype)


@HEAD_REGISTRY.register(HEAD_NAME)
def build_golden_thompson_gibbs_compatibility_heads(
    *, channels: list[int], config: dict
) -> nn.ModuleList:
    """Build one direct Golden--Thompson head per decoder scale."""
    if not isinstance(channels, list) or len(channels) != 4:
        raise ValueError("Golden--Thompson head requires a four-scale channel list")
    num_classes = int(config.get("num_classes", 1))
    unknown = set(config) - {"num_classes"}
    if unknown:
        raise ValueError(f"unknown Golden--Thompson controls: {sorted(unknown)}")
    if num_classes != 1:
        raise ValueError("Golden--Thompson head supports num_classes == 1")
    if any(int(channel) < 1 for channel in channels):
        raise ValueError("all head channels must be positive")
    return nn.ModuleList(
        [
            GoldenThompsonGibbsCompatibilityHead(
                int(channel), num_classes=num_classes
            )
            for channel in channels
        ]
    )


__all__ = [
    "HEAD_NAME",
    "GoldenThompsonGibbsCompatibilityHead",
    "build_golden_thompson_gibbs_compatibility_heads",
    "golden_thompson_gibbs_certificate",
    "symmetric_matrix_from_fields",
    "trace_center_hilbert_schmidt",
]
