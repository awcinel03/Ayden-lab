"""Checksums and git provenance for pipeline runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace) for hashing configs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_sha(*, allow_dirty: bool = False) -> str:
    """Return HEAD's commit SHA.

    Raises ``RuntimeError`` if tracked files have uncommitted changes, unless
    ``allow_dirty`` is set, so every recorded run points at code that exists.
    """
    if not allow_dirty and _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            "Working tree has uncommitted changes; commit first or pass --allow-dirty"
        )
    return _git("rev-parse", "HEAD")
