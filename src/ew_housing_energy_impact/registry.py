"""Artifact registry utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class RegistryEntry:
    timestamp_utc: str
    kind: str
    path: str
    sha256: Optional[str]
    metadata: dict[str, Any]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def register_artifact(
    registry_path: Path,
    kind: str,
    artifact_path: Path,
    metadata: Optional[dict[str, Any]] = None,
    compute_hash: bool = True,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    entry = RegistryEntry(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        kind=kind,
        path=str(artifact_path),
        sha256=sha256_file(artifact_path) if compute_hash and artifact_path.exists() else None,
        metadata=metadata or {},
    )
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
