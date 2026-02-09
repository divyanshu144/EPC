"""Path helpers for locating the repo root and shared config."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    # src/ew_housing_energy_impact/paths.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def load_root_config() -> ModuleType:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import config as root_config  # type: ignore

    return root_config
