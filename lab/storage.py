"""Artifact downloads from the private ``artifacts`` bucket, cached in memory.

Community Cloud disk is ephemeral, so there is no disk cache. Raw datasets are
never read by the app; the ``datasets`` bucket is not readable with the publishable key.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import streamlit as st

from lab.db import TTL_SECONDS, Artifact, _require_client

log = logging.getLogger(__name__)

BUCKET = "artifacts"
MAX_APP_BYTES = 5 * 1024 * 1024  # PLAN §7.5: the app only reads artifacts under 5 MB


@st.cache_data(ttl=TTL_SECONDS, max_entries=32, show_spinner=False)
def _download(storage_path: str) -> bytes:
    return _require_client().storage.from_(BUCKET).download(storage_path)


def artifact_bytes(artifact: Artifact) -> bytes | None:
    """Return an artifact's bytes, or ``None`` if it is too large or unavailable."""
    if artifact.bytes is not None and artifact.bytes > MAX_APP_BYTES:
        log.warning(
            "Artifact %s is %d bytes; over the app limit", artifact.storage_path, artifact.bytes
        )
        return None
    try:
        return _download(artifact.storage_path)
    except Exception:
        log.warning("Artifact download failed: %s", artifact.storage_path, exc_info=True)
        return None


def read_parquet(artifact: Artifact) -> pd.DataFrame | None:
    """Load a Parquet artifact into a DataFrame, or ``None`` if unavailable."""
    data = artifact_bytes(artifact)
    if data is None:
        return None
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception:
        log.warning("Artifact is not valid Parquet: %s", artifact.storage_path, exc_info=True)
        return None
