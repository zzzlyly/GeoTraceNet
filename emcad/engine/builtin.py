from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from emcad.registry import LOSS_REGISTRY, METRIC_REGISTRY


@LOSS_REGISTRY.register("structure_loss_isic18")
def structure_loss(pred, mask, weight: float = 1.0):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (weight * (wbce + wiou)).mean()


@METRIC_REGISTRY.register("dice")
def dice_coefficient(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


@METRIC_REGISTRY.register("iou")
def iou_metric(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def _fast_hd95(pred, gt) -> float:
    pred_np = np.asarray(pred.cpu().numpy(), dtype=bool)
    gt_np = np.asarray(gt.cpu().numpy(), dtype=bool)
    structure = ndi.generate_binary_structure(pred_np.ndim, 1)
    pred_surface = np.logical_xor(pred_np, ndi.binary_erosion(pred_np, structure=structure, border_value=0))
    gt_surface = np.logical_xor(gt_np, ndi.binary_erosion(gt_np, structure=structure, border_value=0))
    if not pred_surface.any() or not gt_surface.any():
        return 100.0
    pred_to_gt = cv2.distanceTransform(
        (~gt_surface).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[pred_surface]
    gt_to_pred = cv2.distanceTransform(
        (~pred_surface).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[gt_surface]
    distances = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(distances, 95))


def binary_metrics(pred, gt, compute_hd95: bool):
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    if not compute_hd95:
        return sensitivity, specificity, precision, float("nan")
    try:
        if pred.sum() > 0 and gt.sum() > 0:
            hd_value = _fast_hd95(pred, gt)
        else:
            hd_value = 100.0
    except Exception:
        hd_value = 100.0
    return sensitivity, specificity, precision, hd_value
