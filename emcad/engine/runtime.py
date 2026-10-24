from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from emcad.config import save_json
from emcad.data.prepare import prepare_isic18_from_config
from emcad.engine import builtin as builtin_engine  # noqa: F401
from emcad.models import builtin_models  # noqa: F401
from emcad.data import isic18 as builtin_datasets  # noqa: F401
from emcad.registry import DATASET_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, MODEL_REGISTRY
from emcad.reporting import (
    build_root_results_csv,
    build_root_summary_csv,
)
from emcad.utils import import_innovation_modules


_PROCESS_RUNTIME_CONFIGURED = False


def configure_process_runtime() -> None:
    global _PROCESS_RUNTIME_CONFIGURED
    if _PROCESS_RUNTIME_CONFIGURED:
        return
    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _PROCESS_RUNTIME_CONFIGURED = True


class AvgMeter:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(value.detach())

    def show(self):
        if not self.values:
            return torch.tensor(0.0)
        return torch.mean(torch.stack(self.values))


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters())


def ensure_runtime_imports(config: dict[str, Any]) -> None:
    runtime_cfg = config.get("runtime", {})
    config_path = config.get("_meta", {}).get("config_path")
    base_dir = Path(config_path).parent if config_path else Path.cwd()
    imports = []
    for item in runtime_cfg.get("imports", []):
        path = Path(item)
        imports.append(str(path if path.is_absolute() else base_dir / path))
    import_innovation_modules(imports)


def build_run_id(config: dict[str, Any], run_index: int, run_seed: int, total_runs: int) -> str:
    exp_name = config.get("experiment", {}).get("name", "")
    if exp_name:
        train_cfg = config.get("train", {})
        explicit_seed_list = train_cfg.get("seed_list")
        has_explicit_seed = explicit_seed_list not in (None, [], "")
        if total_runs == 1 and not has_explicit_seed:
            return exp_name
        return f"{exp_name}_seed{run_seed}"

    model_cfg = config["model"]
    train_cfg = config["train"]
    add = model_cfg.get("add", True)
    aggregation = "add" if add else "concat"
    dw_mode = "parallel" if model_cfg.get("dw_parallel", True) else "series"
    timestamp = time.strftime("%H%M%S")
    run_id = (
        f"{config['dataset'].get('dataset_name', 'dataset')}_{model_cfg.get('encoder', 'model')}"
        f"_EMCAD_kernel_sizes_{model_cfg.get('kernel_sizes', [1, 3, 5])}"
        f"_dw_{dw_mode}_{aggregation}_lgag_ks_{model_cfg.get('lgag_ks', 3)}"
        f"_ef{model_cfg.get('expansion_factor', 2)}_act_{model_cfg.get('activation', 'relu6')}"
        f"_bs{train_cfg.get('batch_size', 16)}_lr{train_cfg.get('lr', 1e-4)}"
        f"_e{train_cfg.get('epochs', 200)}_run{run_index}_seed{run_seed}_t{timestamp}"
    )
    return run_id.replace("[", "").replace("]", "").replace(", ", "_")


def ensure_pretrained_exists(config: dict[str, Any]) -> None:
    model_cfg = config["model"]
    if model_cfg.get("pretrain", True) and "pvt_v2" in model_cfg.get("encoder", ""):
        pretrained_file = os.path.join(
            model_cfg.get("pretrained_dir", "./pretrained_pth/pvt"),
            f"{model_cfg['encoder']}.pth",
        )
        if not os.path.isfile(pretrained_file):
            raise FileNotFoundError(
                f"Required pretrained encoder weights not found: {pretrained_file}"
            )


def build_datamodule(config: dict[str, Any]):
    return DATASET_REGISTRY.build(config["dataset"]["name"], config=config)


def build_model(config: dict[str, Any], device: torch.device, *, for_inference: bool):
    model_config = copy.deepcopy(config)
    if for_inference:
        model_config["model"]["pretrain"] = False
    model = MODEL_REGISTRY.build(model_config["model"]["name"], config=model_config)
    return model.to(device)


def build_optimizer(model, config: dict[str, Any]):
    optimizer_cfg = config["train"].get("optimizer", {})
    name = optimizer_cfg.get("name", "adamw").lower()
    lr = config["train"].get("lr", 1e-4)
    weight_decay = optimizer_cfg.get("weight_decay", 1e-4)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def resolve_loss_fn(config: dict[str, Any]):
    train_cfg = config["train"]
    loss_name = train_cfg.get("loss_name", "structure_loss_isic18")
    loss_target = LOSS_REGISTRY.get(loss_name)
    loss_kwargs = train_cfg.get("loss_kwargs", {})
    if loss_kwargs:
        return partial(loss_target, **loss_kwargs)
    return loss_target


def build_scheduler(optimizer, config: dict[str, Any]):
    scheduler_cfg = config["train"].get("scheduler", {})
    name = scheduler_cfg.get("name", "cosine").lower()
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config["train"].get("epochs", 200),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def clip_gradient(optimizer, grad_clip: float) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def run_model(model, images):
    outputs = model(images)
    return outputs if isinstance(outputs, list) else [outputs]


def compute_training_loss(
    loss_fn,
    outputs: list[torch.Tensor],
    gts: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    apply_mode = config["train"].get("loss_apply", "per_output")
    if apply_mode == "multi_output":
        return loss_fn(outputs, gts)
    if apply_mode != "per_output":
        raise ValueError(f"Unsupported train.loss_apply: {apply_mode}")
    loss_parts = [loss_fn(output, gts) for output in outputs]
    fused_loss = loss_fn(sum(outputs), gts)
    return sum(loss_parts) + fused_loss


def evaluate_split(
    model,
    datamodule,
    split_name: str,
    config: dict[str, Any],
    device: torch.device,
    seed: int | None = None,
    *,
    save_predictions: bool | None = None,
    result_run_name: str | None = None,
    compute_hd95_override: bool | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = datamodule.build_eval_loader(split_name, seed=seed)
    dice_fn = METRIC_REGISTRY.get("dice")
    iou_fn = METRIC_REGISTRY.get("iou")
    compute_hd95 = (
        config["eval"].get("compute_hd95", True)
        if compute_hd95_override is None
        else compute_hd95_override
    )
    save_predictions = (
        config["eval"].get("save_predictions", True)
        if save_predictions is None
        else save_predictions
    )
    use_amp = config["eval"].get("amp", True) and device.type == "cuda"

    model.eval()
    dice_total = 0.0
    iou_total = 0.0
    total_images = 0
    rows: list[dict[str, Any]] = []
    pred_dir = None
    if save_predictions and result_run_name:
        pred_dir = os.path.join(
            config["eval"].get("pred_root", "./predictions_isic18"),
            result_run_name,
            config["dataset"].get("dataset_name", "dataset"),
            split_name,
        )
        os.makedirs(pred_dir, exist_ok=True)

    with torch.inference_mode():
        for batch_idx, (images, gts, original_shapes, names) in enumerate(
            tqdm(loader, desc=f"Inference on {split_name}")
        ):
            images = images.to(device, non_blocking=True)
            gts = gts.float().to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predictions = run_model(model, images)[-1]
            for i in range(len(images)):
                width_orig = int(original_shapes[0][i])
                height_orig = int(original_shapes[1][i])
                pred = predictions[i].unsqueeze(0)
                pred = F.interpolate(
                    pred,
                    size=(height_orig, width_orig),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid().squeeze()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                gt = gts[i].unsqueeze(0)
                gt = F.interpolate(gt, size=(height_orig, width_orig), mode="nearest").squeeze()
                pred_binary = (pred >= 0.5).float()
                gt_binary = (gt >= 0.2).float()

                dice_value = dice_fn(pred_binary, gt_binary).item()
                iou_value = iou_fn(pred_binary, gt_binary).item()
                sensitivity, specificity, precision, hd_value = builtin_engine.binary_metrics(
                    pred_binary, gt_binary, compute_hd95=compute_hd95
                )
                dice_total += dice_value
                iou_total += iou_value
                total_images += 1

                if pred_dir is not None:
                    pred_img = (pred_binary.cpu().numpy() * 255).astype("uint8")
                    cv2.imwrite(os.path.join(pred_dir, names[i]), pred_img)

                rows.append(
                    {
                        "Name": names[i],
                        "Dice": dice_value,
                        "IoU": iou_value,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "Precision": precision,
                        "HD95": hd_value,
                    }
                )
            if config["eval"].get("max_eval_batches", 0) and (batch_idx + 1) >= config["eval"]["max_eval_batches"]:
                break

    if total_images == 0:
        raise RuntimeError(f"No evaluation images found in split: {split_name}")

    summary = {
        "dice": dice_total / total_images,
        "iou": iou_total / total_images,
        "num_images": total_images,
        "pred_dir": pred_dir,
    }
    return summary, rows


def create_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        force=True,
    )


def scheme_name_from_run_name(run_name: str) -> str:
    return run_name.rsplit("_seed", 1)[0] if "_seed" in run_name else run_name


def is_smoke_run_name(run_name: str) -> bool:
    return "_smoke" in run_name or run_name.startswith("smoke_")


def resolve_run_artifact_paths(config: dict[str, Any], run_id: str) -> dict[str, str]:
    train_cfg = config["train"]
    scheme_name = scheme_name_from_run_name(run_id)
    smoke = is_smoke_run_name(run_id)
    save_root = Path(train_cfg.get("save_root", "./model_pth"))
    log_root = Path(train_cfg.get("log_root", "./logs"))
    if smoke:
        model_dir = save_root / "_smoke" / run_id
        log_path = log_root / "_smoke" / f"train_log_{run_id}.log"
    else:
        model_dir = save_root / scheme_name / run_id
        log_path = log_root / scheme_name / f"train_log_{run_id}.log"
    return {
        "scheme_name": scheme_name,
        "model_dir": str(model_dir),
        "log_path": str(log_path),
        "smoke": smoke,
    }


def resolve_existing_run_dir(save_root: str, run_id: str) -> str:
    nested = resolve_run_artifact_paths({"train": {"save_root": save_root, "log_root": "./logs"}}, run_id)["model_dir"]
    legacy = os.path.join(save_root, run_id)
    if os.path.isdir(nested):
        return nested
    return legacy


def cleanup_smoke_artifacts(paths: dict[str, str]) -> None:
    log_path = Path(paths["log_path"])
    model_dir = Path(paths["model_dir"])
    if log_path.exists():
        log_path.unlink()
        parent = log_path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    if model_dir.exists():
        shutil.rmtree(model_dir)
        for parent in model_dir.parents:
            if parent.name in {"model_pth", "logs", ""}:
                break
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            else:
                break


def sync_root_result_exports(config: dict[str, Any], summary_csv: str) -> dict[str, str]:
    report_cfg = config.get("report", {})
    root_results_csv = report_cfg.get("root_results_csv", "./EMCAD_all_test_results.csv")
    root_summary_csv = report_cfg.get("root_summary_csv", "./EMCAD_all_test_summary.csv")
    root_results_path = build_root_results_csv(summary_csv, root_results_csv)
    root_summary_path = build_root_summary_csv(root_results_path, root_summary_csv)
    return {"root_results_csv": root_results_path, "root_summary_csv": root_summary_path}


def migrate_artifact_layout(
    save_root: str,
    log_root: str,
    *,
    drop_smoke: bool = True,
) -> dict[str, int]:
    save_root_path = Path(save_root)
    log_root_path = Path(log_root)
    moved_model_dirs = 0
    moved_logs = 0
    removed_smoke_dirs = 0
    removed_smoke_logs = 0

    if log_root_path.exists():
        for log_path in sorted(log_root_path.glob("train_log_*.log")):
            run_name = log_path.stem.replace("train_log_", "", 1)
            if drop_smoke and is_smoke_run_name(run_name):
                log_path.unlink(missing_ok=True)
                removed_smoke_logs += 1
                continue
            scheme_name = scheme_name_from_run_name(run_name)
            if scheme_name == run_name:
                continue
            target_dir = log_root_path / scheme_name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(log_path), str(target_dir / log_path.name))
            moved_logs += 1
        smoke_log_dir = log_root_path / "_smoke"
        if drop_smoke and smoke_log_dir.exists():
            shutil.rmtree(smoke_log_dir)

    if save_root_path.exists():
        for run_dir in sorted(save_root_path.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            if run_name == "_smoke":
                if drop_smoke:
                    shutil.rmtree(run_dir)
                    removed_smoke_dirs += 1
                continue
            if drop_smoke and is_smoke_run_name(run_name):
                shutil.rmtree(run_dir)
                removed_smoke_dirs += 1
                continue
            scheme_name = scheme_name_from_run_name(run_name)
            if scheme_name == run_name:
                continue
            target_dir = save_root_path / scheme_name / run_name
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(run_dir), str(target_dir))
            moved_model_dirs += 1

    return {
        "moved_model_dirs": moved_model_dirs,
        "moved_logs": moved_logs,
        "removed_smoke_dirs": removed_smoke_dirs,
        "removed_smoke_logs": removed_smoke_logs,
    }


def normalize_summary_checkpoints(summary_csv: str, model_root: str) -> dict[str, int | str]:
    summary_path = Path(summary_csv)
    if not summary_path.exists():
        return {"summary_csv": str(summary_path), "updated_rows": 0}
    df = pd.read_csv(summary_path)
    if "run_id" not in df.columns or "checkpoint" not in df.columns:
        return {"summary_csv": str(summary_path), "updated_rows": 0}

    updated_rows = 0
    for idx, row in df.iterrows():
        run_id = str(row.get("run_id", "") or "")
        if not run_id:
            continue
        checkpoint = str(row.get("checkpoint", "") or "")
        expected = Path(resolve_existing_run_dir(model_root, run_id)) / f"{run_id}-best.pth"
        if not expected.exists():
            continue
        if checkpoint != str(expected) and not Path(checkpoint).exists():
            df.at[idx, "checkpoint"] = str(expected)
            updated_rows += 1

    if updated_rows:
        df.to_csv(summary_path, index=False)
        summary_xlsx = summary_path.with_suffix(".xlsx")
        if summary_xlsx.exists():
            df.to_excel(summary_xlsx, index=False)
    return {"summary_csv": str(summary_path), "updated_rows": updated_rows}


def load_run_metadata(config: dict[str, Any], run_name: str) -> dict[str, Any]:
    model_root = config["eval"].get("model_root", "./model_pth")
    config_path = os.path.join(resolve_existing_run_dir(model_root, run_name), "config.json")
    run_config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
    runtime_info = run_config.get("runtime_info", {})
    seed = runtime_info.get("seed_for_run")
    if seed is None and "_seed" in run_name:
        try:
            seed = int(run_name.rsplit("_seed", 1)[1])
        except ValueError:
            seed = None
    model_cfg = run_config.get("model", config.get("model", {}))
    run_train_cfg = run_config.get("train", config.get("train", {}))
    run_eval_cfg = run_config.get("eval", config.get("eval", {}))
    dataset_cfg = run_config.get("dataset", config.get("dataset", {}))
    return {
        "seed": seed,
        "encoder": model_cfg.get("encoder"),
        "img_size": run_train_cfg.get("img_size"),
        "epoch": run_train_cfg.get("epochs"),
        "batchsize": run_train_cfg.get("batch_size"),
        "test_batchsize": run_eval_cfg.get("batch_size"),
        "deterministic": run_train_cfg.get("deterministic"),
        "train_split": dataset_cfg.get("train_split"),
        "val_split": dataset_cfg.get("val_split"),
        "test_split": dataset_cfg.get("test_split"),
        "run_name": run_config.get("experiment", {}).get("name", run_name),
        "data_root": dataset_cfg.get("data_root"),
    }


def resolve_checkpoint(
    config: dict[str, Any],
    run_id: str | None,
    checkpoint: str | None,
) -> tuple[str, str]:
    if checkpoint:
        checkpoint_path = checkpoint
        run_name = run_id or Path(checkpoint).stem.replace("-best", "").replace("-last", "")
    else:
        if not run_id:
            raise ValueError("Either run_id or checkpoint must be provided.")
        model_root = config["eval"].get("model_root", "./model_pth")
        run_dir = resolve_existing_run_dir(model_root, run_id)
        checkpoint_path = os.path.join(run_dir, f"{run_id}-best.pth")
        run_name = run_id
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path, run_name


def report_results(config: dict[str, Any], summary_csv: str | None = None) -> dict[str, str]:
    configure_process_runtime()
    report_cfg = config.get("report", {})
    root_results_csv = report_cfg.get("root_results_csv", "./EMCAD_all_test_results.csv")
    result_root = config["eval"].get("result_root", "./results_isic18")
    summary_name = report_cfg.get("summary_name", "All_Runs_Summary_ISIC18")
    source_csv = summary_csv or root_results_csv
    legacy_summary_csv = os.path.join(result_root, f"{summary_name}.csv")
    if not os.path.exists(source_csv) and os.path.exists(legacy_summary_csv):
        source_csv = legacy_summary_csv
    root_exports = sync_root_result_exports(config, source_csv)
    print(f"Updated root results CSV: {root_exports['root_results_csv']}")
    print(f"Updated root summary CSV: {root_exports['root_summary_csv']}")
    return {
        "root_results_csv": root_exports["root_results_csv"],
        "root_summary_csv": root_exports["root_summary_csv"],
    }


def maintain_framework(config: dict[str, Any]) -> dict[str, Any]:
    configure_process_runtime()
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    report_cfg = config.get("report", {})
    save_root = train_cfg.get("save_root", "./model_pth")
    log_root = train_cfg.get("log_root", "./logs")
    result_root = eval_cfg.get("result_root", "./results_isic18")
    summary_name = report_cfg.get("summary_name", "All_Runs_Summary_ISIC18")
    root_results_csv = report_cfg.get("root_results_csv", "./EMCAD_all_test_results.csv")
    legacy_summary_csv = os.path.join(result_root, f"{summary_name}.csv")
    summary_csv = root_results_csv if os.path.exists(root_results_csv) else legacy_summary_csv

    migration = migrate_artifact_layout(save_root, log_root, drop_smoke=True)
    normalization = normalize_summary_checkpoints(summary_csv, eval_cfg.get("model_root", save_root))
    report_outputs = report_results(config, summary_csv=summary_csv)
    return {
        "migration": migration,
        "normalization": normalization,
        **report_outputs,
    }
