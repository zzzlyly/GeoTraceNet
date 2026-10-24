from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from emcad.config import save_json
from emcad.utils import configure_reproducibility, resolve_run_seeds

from .runtime import (
    AvgMeter,
    build_datamodule,
    build_model,
    build_optimizer,
    build_run_id,
    build_scheduler,
    cleanup_smoke_artifacts,
    clip_gradient,
    configure_process_runtime,
    count_parameters,
    create_logging,
    ensure_pretrained_exists,
    ensure_runtime_imports,
    evaluate_split,
    resolve_loss_fn,
    resolve_run_artifact_paths,
    run_model,
    compute_training_loss,
)


@dataclass
class EarlyStoppingState:
    """Track validation improvements without changing best-checkpoint selection."""

    enabled: bool
    min_epochs: int
    patience: int
    min_delta: float
    best: float = float("-inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, epoch: int, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.enabled and epoch >= self.min_epochs and self.bad_epochs >= self.patience


def build_early_stopping_state(
    train_cfg: dict[str, Any],
    run_seeds: list[int],
    run_seed: int,
) -> EarlyStoppingState:
    cfg = train_cfg.get("early_stopping", {})
    scope = str(cfg.get("scope", "all")).lower()
    if scope not in {"all", "single_seed0"}:
        raise ValueError(f"Unsupported train.early_stopping.scope: {scope}")
    if str(cfg.get("monitor", "val_dice")).lower() != "val_dice":
        raise ValueError("Only train.early_stopping.monitor=val_dice is supported")
    if str(cfg.get("mode", "max")).lower() != "max":
        raise ValueError("Only train.early_stopping.mode=max is supported")

    patience = int(cfg.get("patience", 30))
    min_epochs = int(cfg.get("min_epochs", 1))
    min_delta = float(cfg.get("min_delta", 0.0))
    if patience <= 0:
        raise ValueError("train.early_stopping.patience must be positive")
    if min_epochs <= 0:
        raise ValueError("train.early_stopping.min_epochs must be positive")
    if min_delta < 0:
        raise ValueError("train.early_stopping.min_delta must be non-negative")

    enabled = bool(cfg.get("enabled", False))
    if scope == "single_seed0":
        enabled = enabled and run_seeds == [0] and run_seed == 0
    return EarlyStoppingState(
        enabled=enabled,
        min_epochs=min_epochs,
        patience=patience,
        min_delta=min_delta,
    )


def train_one_epoch(train_loader, model, optimizer, scaler, epoch: int, config: dict[str, Any], device: torch.device) -> None:
    model.train()
    set_epoch = getattr(train_loader, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
    else:
        dataset = getattr(train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
    loss_record = AvgMeter()
    size_rates = config["train"].get("multi_scale_rates", [0.75, 1.0, 1.25])
    total_step = len(train_loader)
    loss_fn = resolve_loss_fn(config)
    use_amp = config["train"].get("amp", False) and device.type == "cuda"
    amp_dtype = torch.float16

    for step, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images_device = images.to(device, non_blocking=True)
            gts_device = gts.float().to(device, non_blocking=True)
            if rate != 1.0:
                train_size = int(round(config["train"].get("img_size", 352) * rate / 32) * 32)
                images_device = F.interpolate(
                    images_device,
                    size=(train_size, train_size),
                    mode="bilinear",
                    align_corners=True,
                )
                gts_device = F.interpolate(
                    gts_device,
                    size=(train_size, train_size),
                    mode="nearest",
                )

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = run_model(model, images_device)
                loss = compute_training_loss(loss_fn, outputs, gts_device, config)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, config["train"].get("clip", 0.5))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_gradient(optimizer, config["train"].get("clip", 0.5))
                optimizer.step()
            if rate == 1.0:
                loss_record.update(loss)

        if step % config["train"].get("log_interval", 50) == 0 or step == total_step:
            print(
                f"{datetime.now()} Epoch [{epoch:03d}/{config['train'].get('epochs', 200):03d}] "
                f"Step [{step:04d}/{total_step:04d}] "
                f"LR: {optimizer.param_groups[0]['lr']:.8f} "
                f"Loss: {loss_record.show().item():.4f}"
            )
        if config["train"].get("max_train_steps", 0) and step >= config["train"]["max_train_steps"]:
            break


def train_runs(config: dict[str, Any]) -> list[str]:
    configure_process_runtime()
    ensure_runtime_imports(config)
    from emcad.data.prepare import prepare_isic18_from_config

    prepare_isic18_from_config(config)
    ensure_pretrained_exists(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = build_datamodule(config)
    datamodule.validate(require_test=config["train"].get("eval_test_during_train", False))

    train_cfg = config["train"]
    run_seeds = resolve_run_seeds(
        train_cfg.get("seed_list"),
        train_cfg.get("seed", 42),
        train_cfg.get("runs", 1),
    )
    run_ids: list[str] = []

    augmentation_seed_mode = train_cfg.get("augmentation_seed_mode", "legacy_v1")
    if augmentation_seed_mode == "deterministic_v2":
        raise ValueError("deterministic_v2 is unsupported; use deterministic_v3")
    if augmentation_seed_mode not in {"legacy_v1", "deterministic_v3"}:
        raise ValueError(
            "Unsupported train.augmentation_seed_mode: "
            f"{augmentation_seed_mode}; expected legacy_v1 or deterministic_v3"
        )
    if augmentation_seed_mode == "deterministic_v3" and not train_cfg.get("deterministic", True):
        raise ValueError("deterministic_v3 requires train.deterministic=true")

    for offset, run_seed in enumerate(run_seeds):
        run_index = train_cfg.get("run_start", 1) + offset
        configure_reproducibility(
            run_seed,
            train_cfg.get("deterministic", True),
            strict=augmentation_seed_mode == "deterministic_v3",
        )
        run_id = build_run_id(config, run_index, run_seed, len(run_seeds))
        run_ids.append(run_id)
        early_stopping = build_early_stopping_state(train_cfg, run_seeds, run_seed)

        artifact_paths = resolve_run_artifact_paths(config, run_id)
        train_save = artifact_paths["model_dir"]
        os.makedirs(train_save, exist_ok=True)
        log_path = artifact_paths["log_path"]
        create_logging(log_path)

        model = build_model(config, device, for_inference=False)
        param_count = count_parameters(model)
        print(f"Run ID: {run_id}")
        print(f"Encoder: {config['model'].get('encoder')} | Decoder: EMCAD | Params: {param_count}")
        logging.info("Run ID: %s", run_id)
        logging.info("Parameter count: %d", param_count)
        logging.info("Deterministic: %s | Seed: %d", train_cfg.get("deterministic", True), run_seed)

        resolved_config = copy.deepcopy(config)
        resolved_config.setdefault("runtime_info", {})
        resolved_config["runtime_info"].update(
            {
                "run_id": run_id,
                "seed_for_run": run_seed,
                "resolved_seed_list": run_seeds,
                "device": str(device),
                "early_stopping_active": early_stopping.enabled,
                "reproducibility_protocol": augmentation_seed_mode,
                "augmentation_seed_mode": augmentation_seed_mode,
                "strict_deterministic_algorithms": augmentation_seed_mode == "deterministic_v3",
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "augmentation_seed_derivation": "blake2s-64(base_seed,epoch,sorted_sample_index,sample_filename)"
                if augmentation_seed_mode == "deterministic_v3"
                else None,
            }
        )
        save_json(resolved_config, os.path.join(train_save, "config.json"))

        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        scaler = GradScaler("cuda", enabled=train_cfg.get("amp", False) and device.type == "cuda")
        train_loader = datamodule.build_train_loader(seed=run_seed)

        best_val = 0.0
        best_epoch = 0
        test_at_best_val = None
        total_train_time = 0.0
        epochs_completed = 0
        stopped_early = False

        for epoch in range(1, train_cfg.get("epochs", 200) + 1):
            epochs_completed = epoch
            epoch_start = time.time()
            prev_scale = scaler.get_scale() if scaler.is_enabled() else None
            train_one_epoch(train_loader, model, optimizer, scaler, epoch, config, device)
            if not scaler.is_enabled() or scaler.get_scale() >= prev_scale:
                scheduler.step()
            total_train_time += time.time() - epoch_start

            last_path = os.path.join(train_save, f"{run_id}-last.pth")
            torch.save(model.state_dict(), last_path)

            val_split = datamodule.split_name("val")
            val_metrics, _ = evaluate_split(
                model,
                datamodule,
                val_split,
                config,
                device,
                seed=run_seed,
                compute_hd95_override=train_cfg.get("eval_compute_hd95", False),
            )
            test_metrics = None
            if train_cfg.get("eval_test_during_train", False):
                test_split = datamodule.split_name("test")
                test_metrics, _ = evaluate_split(
                    model,
                    datamodule,
                    test_split,
                    config,
                    device,
                    seed=run_seed,
                    compute_hd95_override=train_cfg.get("eval_compute_hd95", False),
                )
                logging.info(
                    "Epoch %d | %s Dice %.4f IoU %.4f | %s Dice %.4f IoU %.4f",
                    epoch,
                    val_split,
                    val_metrics["dice"],
                    val_metrics["iou"],
                    test_split,
                    test_metrics["dice"],
                    test_metrics["iou"],
                )
                print(
                    f"Epoch: {epoch}, {val_split}: Dice {val_metrics['dice']:.4f}, "
                    f"IoU {val_metrics['iou']:.4f} | {test_split}: Dice {test_metrics['dice']:.4f}, "
                    f"IoU {test_metrics['iou']:.4f}"
                )
            else:
                logging.info(
                    "Epoch %d | %s Dice %.4f IoU %.4f",
                    epoch,
                    val_split,
                    val_metrics["dice"],
                    val_metrics["iou"],
                )
                print(
                    f"Epoch: {epoch}, {val_split}: Dice {val_metrics['dice']:.4f}, "
                    f"IoU {val_metrics['iou']:.4f}"
                )

            if val_metrics["dice"] > best_val:
                best_val = val_metrics["dice"]
                best_epoch = epoch
                test_at_best_val = test_metrics["dice"] if test_metrics is not None else None
                best_path = os.path.join(train_save, f"{run_id}-best.pth")
                torch.save(model.state_dict(), best_path)
                logging.info("Saved best checkpoint: val dice improved to %.4f", best_val)
                print(f"Saved best checkpoint with {val_split} dice {best_val:.4f}")

            if early_stopping.update(epoch, val_metrics["dice"]):
                stopped_early = True
                message = (
                    f"Early stopping at epoch {epoch}: {early_stopping.bad_epochs} consecutive "
                    f"validation checks without an improvement greater than "
                    f"{early_stopping.min_delta:g}; patience={early_stopping.patience}, "
                    f"min_epochs={early_stopping.min_epochs}."
                )
                print(message)
                logging.info(message)
                break

        test_summary = (
            f"{datamodule.split_name('test')} Dice at Best {datamodule.split_name('val')}: "
            f"{test_at_best_val:.4f}\n"
            if test_at_best_val is not None
            else f"{datamodule.split_name('test')} evaluation during train: disabled\n"
        )
        summary = (
            f"\n{'=' * 40}\nFINAL RESULTS: {run_id}\n"
            f"Best {datamodule.split_name('val')} Dice: {best_val:.4f}\n"
            f"Best Epoch: {best_epoch}\n"
            f"Epochs Completed: {epochs_completed}\n"
            f"Stopped Early: {stopped_early}\n"
            f"{test_summary}"
            f"Total Train Time: {total_train_time:.2f}s\n{'=' * 40}"
        )
        print(summary)
        logging.info(summary)

        if artifact_paths["smoke"]:
            cleanup_smoke_artifacts(artifact_paths)

    return run_ids


def smoke_run(config: dict[str, Any]) -> list[str]:
    configure_process_runtime()
    smoke_config = copy.deepcopy(config)
    exp_name = smoke_config.get("experiment", {}).get("name", "experiment")
    smoke_config.setdefault("experiment", {})
    smoke_config["experiment"]["name"] = f"{exp_name}_smoke"
    smoke_config["train"]["epochs"] = 1
    smoke_config["train"]["batch_size"] = 1
    smoke_config["train"]["num_workers"] = 0
    smoke_config["train"]["seed_list"] = [smoke_config["train"].get("seed", 42)]
    smoke_config["train"]["max_train_steps"] = 1
    smoke_config["eval"]["batch_size"] = 1
    smoke_config["eval"]["num_workers"] = 0
    smoke_config["eval"]["max_eval_batches"] = 1
    smoke_config["eval"]["compute_hd95"] = False
    smoke_config["eval"]["save_predictions"] = False
    return train_runs(smoke_config)


def ablate_runs(config: dict[str, Any]) -> list[str]:
    return train_runs(config)
