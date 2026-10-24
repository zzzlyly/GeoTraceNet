from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def import_innovation_modules(imports: list[str] | None) -> None:
    for item in imports or []:
        if not item:
            continue
        if item.endswith(".py") or "/" in item:
            module_path = Path(item).resolve()
            if not module_path.is_file():
                raise FileNotFoundError(f"Innovation module file not found: {module_path}")
            module_name = f"emcad_ext_{module_path.stem}"
            existing = sys.modules.get(module_name)
            if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == module_path:
                continue
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load innovation module: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(item)
