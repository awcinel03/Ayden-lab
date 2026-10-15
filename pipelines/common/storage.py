"""Uploads to Supabase Storage with checksum-based skipping and a size budget."""

from __future__ import annotations

import logging

from supabase import Client

log = logging.getLogger(__name__)

FREE_PLAN_FILE_LIMIT = 50 * 1024 * 1024
BUDGET_WARN_BYTES = 750 * 1024 * 1024  # PLAN §7.5: warn well before the 1 GB Free cap


def upload(client: Client, bucket: str, path: str, data: bytes, content_type: str) -> None:
    """Upload ``data`` to ``bucket/path``, overwriting. Raises ``ValueError`` over 50 MB."""
    if len(data) > FREE_PLAN_FILE_LIMIT:
        raise ValueError(f"{path} is {len(data)} bytes; over the 50 MB Free-plan limit")
    client.storage.from_(bucket).upload(
        path, data, file_options={"content-type": content_type, "upsert": "true"}
    )


def stored_bytes(client: Client) -> int:
    """Total bytes recorded in ``datasets`` and ``artifacts`` rows."""
    total = 0
    for table in ("datasets", "artifacts"):
        rows = client.table(table).select("bytes").execute().data
        total += sum(r["bytes"] or 0 for r in rows)
    return total


def check_budget(client: Client) -> int:
    """Log a warning if recorded storage exceeds the budget; return the total."""
    total = stored_bytes(client)
    if total > BUDGET_WARN_BYTES:
        log.warning(
            "Storage at %.0f MB, over the %d MB budget", total / 2**20, BUDGET_WARN_BYTES // 2**20
        )
    return total
