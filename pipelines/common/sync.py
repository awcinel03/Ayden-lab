"""Upsert ``experiments`` rows from ``lab.registry``, the single source of truth."""

from __future__ import annotations

from lab.registry import ENTRIES, Entry
from supabase import Client


def experiment_row(entry: Entry) -> dict[str, object]:
    """Map a registry entry to an ``experiments`` row."""
    return {
        "slug": entry.slug,
        "title": entry.title,
        "category": entry.category.key,
        "description": entry.summary,
        "published": entry.published,
    }


def sync_experiments(client: Client, entries: tuple[Entry, ...] = ENTRIES) -> dict[str, int]:
    """Upsert every registry entry on ``slug`` and return ``{slug: experiment id}``."""
    if not entries:
        return {}
    rows = (
        client.table("experiments")
        .upsert([experiment_row(e) for e in entries], on_conflict="slug")
        .execute()
        .data
    )
    return {r["slug"]: r["id"] for r in rows}
