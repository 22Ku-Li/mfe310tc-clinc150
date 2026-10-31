"""Lightweight configuration helpers that do not require deep-learning packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data import PROJECT_ROOT


def load_config(path: Path) -> dict[str, Any]:
    """Load an experiment YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(value: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
