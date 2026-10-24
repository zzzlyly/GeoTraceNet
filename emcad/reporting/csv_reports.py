from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import re
import tempfile
from pathlib import Path

import pandas as pd


ROOT_RESULTS_COLUMNS = [
    "scheme",
    "dataset",
    "split",
    "seed",
    "run_id",
    "dice",
    "iou",
    "sensitivity",
    "specificity",
    "precision",
    "HD95",
    "run_name",
    "encoder",
    "img_size",
    "batchsize",
    "test_batchsize",
    "epoch",
    "deterministic",
    "compute_hd95",
    "save_predictions",
    "checkpoint",
    "evaluated_at",
]


def write_result_table(df: pd.DataFrame, result_root: str, file_stem: str) -> tuple[str, str]:
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{file_stem}.csv"
    xlsx_path = root / f"{file_stem}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return str(csv_path), str(xlsx_path)


def update_summary_files(summary_row: dict, result_root: str, summary_name: str) -> tuple[str, str]:
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{summary_name}.csv"
    xlsx_path = root / f"{summary_name}.xlsx"
    dedupe_keys = ["run_id", "checkpoint", "dataset", "split"]
    new_df = pd.DataFrame([summary_row])

    if csv_path.exists():
        existing_csv = pd.read_csv(csv_path)
        combined_csv = pd.concat([existing_csv, new_df], ignore_index=True)
        combined_csv = combined_csv.drop_duplicates(subset=dedupe_keys, keep="last")
    else:
        combined_csv = new_df
    combined_csv.to_csv(csv_path, index=False)

    if xlsx_path.exists():
        existing_xlsx = pd.read_excel(xlsx_path)
        combined_xlsx = pd.concat([existing_xlsx, new_df], ignore_index=True)
        combined_xlsx = combined_xlsx.drop_duplicates(subset=dedupe_keys, keep="last")
    else:
        combined_xlsx = new_df
    combined_xlsx.to_excel(xlsx_path, index=False)

    return str(csv_path), str(xlsx_path)


def _ensure_scheme_column(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    run_id_series = normalized.get("run_id", pd.Series("", index=normalized.index)).fillna("").astype(str)
    run_name_series = normalized.get("run_name", pd.Series("", index=normalized.index)).fillna("").astype(str)
    scheme = run_name_series.mask(run_name_series == "", run_id_series)
    scheme = scheme.str.replace(r"_seed\d+$", "", regex=True)
    normalized["scheme"] = scheme
    return normalized


def _is_smoke_name(name: str) -> bool:
    return "_smoke" in name or name.startswith("smoke_")


def _filter_root_export_rows(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _ensure_scheme_column(df)
    if "split" in normalized.columns:
        normalized = normalized[normalized["split"].astype(str) == "test"]
    return normalized[~normalized["scheme"].astype(str).apply(_is_smoke_name)].copy()


def _format_seed_series(items: pd.Series) -> tuple[int, str]:
    cleaned = []
    for item in items.dropna().tolist():
        try:
            cleaned.append(int(item))
        except (TypeError, ValueError):
            continue
    unique_sorted = sorted(set(cleaned))
    return len(unique_sorted), ",".join(str(item) for item in unique_sorted)


def _prepare_root_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    filtered = _filter_root_export_rows(df)
    dedupe_keys = [key for key in ["run_id", "checkpoint", "dataset", "split"] if key in filtered.columns]
    if dedupe_keys:
        filtered = filtered.drop_duplicates(subset=dedupe_keys, keep="last")
    if "seed" in filtered.columns:
        filtered["seed"] = pd.to_numeric(filtered["seed"], errors="coerce").astype("Int64")
    columns = [item for item in ROOT_RESULTS_COLUMNS if item in filtered.columns]
    export_df = filtered[columns].copy()
    sort_columns = [item for item in ["scheme", "dataset", "split", "seed", "run_id"] if item in export_df.columns]
    if sort_columns:
        export_df = export_df.sort_values(sort_columns, na_position="last")
    return export_df.reset_index(drop=True)


@contextmanager
def _csv_file_lock(csv_path: Path):
    lock_root = Path(tempfile.gettempdir()) / "emcad_csv_locks"
    lock_name = hashlib.sha1(str(csv_path.resolve()).encode("utf-8")).hexdigest() + ".lock"
    lock_path = lock_root / lock_name
    lock_root.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def update_root_results_file(summary_row: dict, output_csv: str) -> str:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _csv_file_lock(output_path):
        new_df = pd.DataFrame([summary_row])
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        else:
            combined_df = new_df
        export_df = _prepare_root_results_frame(combined_df)
        export_df.to_csv(output_path, index=False)
    return str(output_path)


def update_root_summary_file(root_results_csv: str, output_csv: str) -> str:
    # Keep the source lock until the derived summary is committed.
    summary_path = Path(root_results_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _csv_file_lock(summary_path):
        grouped = _build_multiseed_summary_frame(_filter_root_export_rows(pd.read_csv(summary_path)))
        with _csv_file_lock(output_path):
            grouped.to_csv(output_path, index=False)
    return str(output_path)


def build_comparison_csv(summary_csv: str, output_csv: str) -> str:
    df = _ensure_scheme_column(pd.read_csv(summary_csv))
    preferred = [
        "scheme",
        "run_id",
        "run_name",
        "seed",
        "dataset",
        "split",
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "precision",
        "HD95",
        "encoder",
        "img_size",
        "batchsize",
        "test_batchsize",
        "epoch",
        "deterministic",
        "compute_hd95",
        "save_predictions",
        "checkpoint",
        "evaluated_at",
    ]
    columns = [item for item in preferred if item in df.columns]
    comparison = (
        df[columns]
        .sort_values(["scheme", "dataset", "split", "seed", "run_id"], na_position="last")
        .reset_index(drop=True)
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)
    return output_csv


def _build_multiseed_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_scheme_column(df)
    metrics = ["dice", "iou", "sensitivity", "specificity", "precision", "HD95"]
    available_metrics = [metric for metric in metrics if metric in df.columns]
    group_keys = ["scheme", "dataset", "split"]
    if df.empty:
        ordered_columns = ["scheme", "run_name", "dataset", "split", "seed_count", "seeds", "run_count"]
        for metric in available_metrics:
            for suffix in ("mean", "max", "std", "min"):
                ordered_columns.append(f"{metric}_{suffix}")
        return pd.DataFrame(columns=ordered_columns)
    grouped = (
        df.groupby(group_keys, dropna=False)[available_metrics]
        .agg(["mean", "std", "max", "min"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part).strip("_")
        for col in grouped.columns.to_flat_index()
    ]
    seed_df = (
        df.groupby(group_keys, dropna=False)["seed"]
        .apply(_format_seed_series)
        .reset_index(name="seed_info")
    )
    seed_df[["seed_count", "seeds"]] = pd.DataFrame(seed_df["seed_info"].tolist(), index=seed_df.index)
    seed_df = seed_df.drop(columns=["seed_info"])
    run_count_df = (
        df.groupby(group_keys, dropna=False)["run_id"]
        .nunique()
        .reset_index(name="run_count")
    )
    grouped = grouped.merge(seed_df, on=group_keys, how="left")
    grouped = grouped.merge(run_count_df, on=group_keys, how="left")
    grouped["run_name"] = grouped["scheme"]

    ordered_columns = ["scheme", "run_name", "dataset", "split", "seed_count", "seeds", "run_count"]
    for metric in available_metrics:
        for suffix in ("mean", "max", "std", "min"):
            column = f"{metric}_{suffix}"
            if column in grouped.columns:
                ordered_columns.append(column)
    remaining = [column for column in grouped.columns if column not in ordered_columns]
    grouped = grouped[ordered_columns + remaining]
    sort_columns = ["dataset", "split", "scheme"]
    ascending = [True, True, True]
    if "dice_mean" in grouped.columns:
        sort_columns = ["dataset", "split", "dice_mean", "scheme"]
        ascending = [True, True, False, True]
    return grouped.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)


def build_multiseed_summary(summary_csv: str, output_csv: str) -> str:
    grouped = _build_multiseed_summary_frame(pd.read_csv(summary_csv))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)
    return output_csv


def build_root_results_csv(summary_csv: str, output_csv: str) -> str:
    export_df = _prepare_root_results_frame(pd.read_csv(summary_csv))
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _csv_file_lock(output_path):
        export_df.to_csv(output_path, index=False)
    return output_csv


def build_root_summary_csv(summary_csv: str, output_csv: str) -> str:
    return update_root_summary_file(summary_csv, output_csv)
