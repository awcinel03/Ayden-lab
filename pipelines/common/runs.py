"""Run lifecycle: fingerprinting, idempotent start, results and artifact records."""

from __future__ import annotations

import datetime
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from pipelines.common.provenance import canonical_json, sha256_bytes
from supabase import Client

RESULTS_BATCH = 500


def fingerprint(config: dict[str, Any], git_sha: str, dataset_sha256s: Sequence[str]) -> str:
    """Idempotency key of a run: same config, code and input data ⇒ same fingerprint."""
    payload = {"config": config, "git_sha": git_sha, "datasets": sorted(dataset_sha256s)}
    return sha256_bytes(canonical_json(payload).encode())


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def start_run(
    client: Client,
    *,
    experiment_id: int,
    config: dict[str, Any],
    git_sha: str,
    datasets: dict[int, str],
) -> int | None:
    """Start a run, or return ``None`` if an identical run already succeeded.

    ``datasets`` maps dataset id to its sha256. A previous failed or interrupted
    run with the same fingerprint is reused and its results and artifacts cleared.
    """
    fp = fingerprint(config, git_sha, list(datasets.values()))
    existing = client.table("runs").select("id, status").eq("fingerprint", fp).execute().data
    if existing and existing[0]["status"] == "succeeded":
        return None
    if existing:
        run_id = existing[0]["id"]
        client.table("results").delete().eq("run_id", run_id).execute()
        client.table("artifacts").delete().eq("run_id", run_id).execute()
        client.table("runs").update(
            {"status": "running", "started_at": _now(), "finished_at": None}
        ).eq("id", run_id).execute()
    else:
        run_id = (
            client.table("runs")
            .insert(
                {
                    "experiment_id": experiment_id,
                    "status": "running",
                    "config": config,
                    "git_sha": git_sha,
                    "fingerprint": fp,
                }
            )
            .execute()
            .data[0]["id"]
        )
    if datasets:
        client.table("run_datasets").upsert(
            [{"run_id": run_id, "dataset_id": d} for d in datasets],
            on_conflict="run_id,dataset_id",
        ).execute()
    return run_id


def finish_run(client: Client, run_id: int, *, succeeded: bool) -> None:
    """Mark a run succeeded or failed and stamp ``finished_at``."""
    client.table("runs").update(
        {"status": "succeeded" if succeeded else "failed", "finished_at": _now()}
    ).eq("id", run_id).execute()


@contextmanager
def tracked(client: Client, run_id: int) -> Iterator[int]:
    """Context manager that marks the run failed on exception, succeeded otherwise."""
    try:
        yield run_id
    except BaseException:
        finish_run(client, run_id, succeeded=False)
        raise
    finish_run(client, run_id, succeeded=True)


def write_results(client: Client, run_id: int, rows: Sequence[dict[str, Any]]) -> None:
    """Upsert long-format results (``metric``, ``value``, optional ``unit`` and ``dims``)."""
    payload = [
        {
            "run_id": run_id,
            "metric": r["metric"],
            "value": r["value"],
            "unit": r.get("unit"),
            "dims": r.get("dims", {}),
        }
        for r in rows
    ]
    for i in range(0, len(payload), RESULTS_BATCH):
        client.table("results").upsert(
            payload[i : i + RESULTS_BATCH], on_conflict="run_id,metric,dims"
        ).execute()


def record_artifact(
    client: Client,
    run_id: int,
    *,
    kind: str,
    storage_path: str,
    data: bytes,
    content_type: str,
) -> None:
    """Upsert the ``artifacts`` row for a file already uploaded to ``storage_path``."""
    client.table("artifacts").upsert(
        {
            "run_id": run_id,
            "kind": kind,
            "storage_path": storage_path,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "content_type": content_type,
        },
        on_conflict="storage_path",
    ).execute()
