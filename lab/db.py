"""Read-only access to Supabase for the app, using the publishable key.

Public helpers never raise: on missing secrets, network or API errors they log a
warning and return ``None`` or an empty frame, so pages can show ``st.info``.
Failures are not cached (``st.cache_data`` does not cache exceptions).
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from supabase import Client, create_client

log = logging.getLogger(__name__)

TTL_SECONDS = 3600
_PUBLISHABLE_KEY_RE = re.compile(r"sb_publishable_[A-Za-z0-9_-]+")
PAGE_SIZE = 1000  # PostgREST max_rows; larger reads are paged


@dataclass(frozen=True)
class Run:
    """A succeeded run of an experiment."""

    id: int
    experiment_slug: str
    started_at: datetime.datetime
    finished_at: datetime.datetime | None
    config: dict[str, Any]
    git_sha: str


@dataclass(frozen=True)
class Dataset:
    """Provenance record for a dataset used by a run."""

    slug: str
    name: str
    source: str
    source_url: str
    license: str | None
    citation: str | None
    sha256: str
    retrieved_at: datetime.datetime
    row_count: int | None


@dataclass(frozen=True)
class Artifact:
    """A file a run stored in the ``artifacts`` bucket."""

    kind: str
    storage_path: str
    bytes: int | None
    content_type: str | None


@st.cache_resource
def get_client() -> Client | None:
    """Return a cached Supabase client, or ``None`` if secrets are not configured."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_PUBLISHABLE_KEY"]
    except (KeyError, FileNotFoundError):
        log.warning("Supabase secrets not configured; data will be unavailable")
        return None
    if not publishable_key_looks_valid(key):
        log.warning(
            "SUPABASE_PUBLISHABLE_KEY looks like a placeholder or is truncated (expected "
            "'sb_publishable_' followed by ASCII letters, digits, '-' or '_'); "
            "data will be unavailable"
        )
        return None
    return create_client(url, key)


def publishable_key_looks_valid(key: object) -> bool:
    """True if ``key`` has the publishable-key format. Never logs or returns the value."""
    return isinstance(key, str) and _PUBLISHABLE_KEY_RE.fullmatch(key) is not None


def _require_client() -> Client:
    client = get_client()
    if client is None:
        raise RuntimeError("Supabase client unavailable")
    return client


def _safe[T](fn: Callable[..., T], *args: Any, default: T) -> T:
    try:
        return fn(*args)
    except Exception:
        log.warning("Supabase read failed: %s%r", fn.__name__, args, exc_info=True)
        return default


def _ts(value: str | None) -> datetime.datetime | None:
    return datetime.datetime.fromisoformat(value) if value else None


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _latest_run(experiment_slug: str) -> Run | None:
    rows = (
        _require_client()
        .table("latest_runs")
        .select("id, experiment_slug, started_at, finished_at, config, git_sha")
        .eq("experiment_slug", experiment_slug)
        .limit(1)
        .execute()
        .data
    )
    if not rows:
        return None
    r = rows[0]
    return Run(
        id=r["id"],
        experiment_slug=r["experiment_slug"],
        started_at=_ts(r["started_at"]),
        finished_at=_ts(r["finished_at"]),
        config=r["config"],
        git_sha=r["git_sha"],
    )


def latest_run(experiment_slug: str) -> Run | None:
    """Latest succeeded run of a published experiment, or ``None``."""
    return _safe(_latest_run, experiment_slug, default=None)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _results(run_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        page = (
            _require_client()
            .table("results")
            .select("metric, value, unit, dims")
            .eq("run_id", run_id)
            .order("id")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data
        )
        rows += page
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "dims"])


def results(run_id: int) -> pd.DataFrame:
    """All results of a run in long format (metric, value, unit, dims); empty on failure."""
    return _safe(
        _results, run_id, default=pd.DataFrame(columns=["metric", "value", "unit", "dims"])
    )


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _run_datasets(run_id: int) -> list[Dataset]:
    rows = (
        _require_client()
        .table("run_datasets")
        .select(
            "datasets(slug, name, source, source_url, license, citation, sha256,"
            " retrieved_at, row_count)"
        )
        .eq("run_id", run_id)
        .execute()
        .data
    )
    out = []
    for row in rows:
        d = row["datasets"]
        out.append(Dataset(**{**d, "retrieved_at": _ts(d["retrieved_at"])}))
    return out


def run_datasets(run_id: int) -> list[Dataset]:
    """Datasets a run used; empty on failure."""
    return _safe(_run_datasets, run_id, default=[])


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _artifacts(run_id: int) -> list[Artifact]:
    rows = (
        _require_client()
        .table("artifacts")
        .select("kind, storage_path, bytes, content_type")
        .eq("run_id", run_id)
        .order("storage_path")
        .execute()
        .data
    )
    return [Artifact(**r) for r in rows]


def artifacts(run_id: int) -> list[Artifact]:
    """Artifacts a run stored; empty on failure."""
    return _safe(_artifacts, run_id, default=[])
