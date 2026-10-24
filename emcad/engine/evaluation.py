from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import torch

from .runtime import (
    build_datamodule,
    build_model,
    configure_process_runtime,
    ensure_runtime_imports,
    evaluate_split,
    load_run_metadata,
    resolve_checkpoint,
)
from emcad.reporting import update_root_results_file, update_root_summary_file


def evaluate_run(
    config: dict[str, Any],
    *,
    run_id: str | None = None,
    checkpoint: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    configure_process_runtime()
    ensure_runtime_imports(config)
    from emcad.data.prepare import prepare_isic18_from_config

    prepare_isic18_from_config(config)
    datamodule = build_datamodule(config)
    checkpoint_path, run_name = resolve_checkpoint(config, run_id, checkpoint)
    run_metadata = load_run_metadata(config, run_name)

    split_name = split or datamodule.split_name("test")
    if not datamodule.split_exists(split_name):
        raise FileNotFoundError(f"Missing split directories: {datamodule.split_root(split_name)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device, for_inference=True)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    metrics, rows = evaluate_split(
        model,
        datamodule,
        split_name,
        config,
        device,
        seed=run_metadata.get("seed"),
        result_run_name=run_name,
    )

    df = pd.DataFrame(rows)
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row["Name"] = "AVERAGE"
    dataset_name = config["dataset"].get("dataset_name", "dataset")

    summary_row = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_name,
        "checkpoint": checkpoint_path,
        "dataset": dataset_name,
        "split": split_name,
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "sensitivity": mean_row["Sensitivity"],
        "specificity": mean_row["Specificity"],
        "precision": mean_row["Precision"],
        "HD95": mean_row.get("HD95", float("nan")),
        "compute_hd95": config["eval"].get("compute_hd95", True),
        "save_predictions": config["eval"].get("save_predictions", True),
    }
    summary_row.update(run_metadata)
    report_cfg = config.get("report", {})
    root_results_csv = report_cfg.get("root_results_csv", "./EMCAD_all_test_results.csv")
    root_summary_csv = report_cfg.get("root_summary_csv", "./EMCAD_all_test_summary.csv")
    root_results_path = update_root_results_file(summary_row, root_results_csv)
    root_summary_path = update_root_summary_file(root_results_path, root_summary_csv)

    print(
        f"Evaluation complete. Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}"
    )
    if metrics.get("pred_dir"):
        print(f"Saved predictions to: {metrics['pred_dir']}")
    print(f"Updated root results CSV: {root_results_path}")
    print(f"Updated root summary CSV: {root_summary_path}")

    return {
        "result_csv": root_results_path,
        "result_xlsx": None,
        "summary_csv": root_results_path,
        "summary_xlsx": None,
        "summary_row": summary_row,
        "root_results_csv": root_results_path,
        "root_summary_csv": root_summary_path,
    }
