from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            key = name or getattr(obj, "__name__", None)
            if not key:
                raise ValueError(f"Registry {self.name} received an unnamed object.")
            if key in self._items:
                raise KeyError(f"{key} is already registered in {self.name}.")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"{name} is not registered in {self.name}. Available: {available}"
            )
        return self._items[name]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        target = self.get(name)
        if isinstance(target, type):
            return target(*args, **kwargs)
        if callable(target):
            return target(*args, **kwargs)
        return target

    def keys(self) -> list[str]:
        return sorted(self._items)
