from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from emcad.utils.dicts import deep_merge, set_by_dotted_path


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return data


def _resolve_config(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Detected recursive config include: {path}")
    seen.add(path)
    data = _load_yaml_file(path)
    includes = data.pop("includes", [])
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (path.parent / include).resolve()
        merged = deep_merge(merged, _resolve_config(include_path, seen))
    merged = deep_merge(merged, data)
    merged.setdefault("_meta", {})
    merged["_meta"]["config_path"] = str(path)
    seen.remove(path)
    return merged


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override, expected key=value: {override}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_by_dotted_path(config, key, value)
    return config


def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    config = _resolve_config(Path(config_path), seen=set())
    return apply_overrides(config, overrides)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
