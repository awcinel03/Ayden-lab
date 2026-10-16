"""Ingest StatsBomb open-data JSON files: download → checksum → upload to Storage → upsert row.

Run: python -m pipelines.ingest.statsbomb pl-2003-04-matches
     python -m pipelines.ingest.statsbomb --all

Idempotent: if a downloaded file's sha256 matches the recorded row and the
object exists in Storage, nothing is written. Files are stored gzipped (the
bucket does not accept JSON); the recorded sha256 is of the original JSON.
Files are mirrored to the private ``datasets`` bucket only; the StatsBomb Public
Data User Agreement forbids redistributing the data, so the app never reads them
(it shows derived results).
One request per file, with a descriptive User-Agent.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from pipelines.common.client import get_client
from pipelines.common.provenance import sha256_bytes
from pipelines.common.storage import check_budget, upload
from pipelines.ingest.snap import download
from supabase import Client

log = logging.getLogger(__name__)

BUCKET = "datasets"
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SOURCE_PAGE = "https://github.com/statsbomb/open-data"
LICENSE = (
    "StatsBomb Public Data User Agreement (2023-09-08): non-commercial research use; no "
    "redistribution of the data; publications must credit StatsBomb with its logo"
)
CITATION = "StatsBomb Open Data. StatsBomb Services Ltd. https://github.com/statsbomb/open-data"


@dataclass(frozen=True)
class StatsBombSource:
    """One StatsBomb open-data JSON file and its provenance metadata."""

    slug: str
    name: str
    path: str  # under data/, e.g. "events/3749079.json"
    notes: str

    @property
    def url(self) -> str:
        """Raw download URL on GitHub."""
        return f"{BASE_URL}/{self.path}"


def _match_sources(match_id: int, label: str) -> dict[str, StatsBombSource]:
    return {
        f"events-{match_id}": StatsBombSource(
            slug=f"statsbomb-events-{match_id}",
            name=f"StatsBomb events: {label}",
            path=f"events/{match_id}.json",
            notes="JSON array of every on-ball event in the match (passes, carries, shots, …).",
        ),
        f"lineups-{match_id}": StatsBombSource(
            slug=f"statsbomb-lineups-{match_id}",
            name=f"StatsBomb lineups: {label}",
            path=f"lineups/{match_id}.json",
            notes="JSON array of the two teams' squads, with player names and nicknames.",
        ),
    }


SOURCES: dict[str, StatsBombSource] = {
    "pl-2003-04-matches": StatsBombSource(
        slug="statsbomb-pl-2003-04-matches",
        name="StatsBomb matches: Premier League 2003/04",
        path="matches/2/44.json",
        notes="JSON array of the season's matches in the open data (Arsenal's 38 league games).",
    ),
    **_match_sources(3749079, "Arsenal 2-1 Chelsea, 2003-10-18"),
    **_match_sources(3749552, "Manchester United 0-0 Arsenal, 2003-09-21"),
}


def count_records(data: bytes) -> int:
    """Number of top-level records in a JSON array file."""
    return len(json.loads(data))


def ingest(
    client: Client, source: StatsBombSource, fetch: Callable[[str], bytes] = download
) -> str:
    """Mirror ``source`` into Storage and upsert its row.

    Returns ``"unchanged"`` if the file matches the recorded sha256, else ``"updated"``.
    """
    data = fetch(source.url)
    sha = sha256_bytes(data)
    path = f"statsbomb/{source.path}.gz"
    rows = client.table("datasets").select("sha256").eq("slug", source.slug).execute().data
    if rows and rows[0]["sha256"] == sha and client.storage.from_(BUCKET).exists(path):
        return "unchanged"
    blob = gzip.compress(data, mtime=0)
    upload(client, BUCKET, path, blob, "application/gzip")
    client.table("datasets").upsert(
        {
            "slug": source.slug,
            "name": source.name,
            "source": "StatsBomb Open Data",
            "source_url": SOURCE_PAGE,
            "license": LICENSE,
            "citation": CITATION,
            "storage_path": f"{BUCKET}/{path}",
            "format": "json.gz",
            "bytes": len(blob),
            "sha256": sha,
            "row_count": count_records(data),
            "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "notes": f"{source.notes} Downloaded from {source.url}.",
        },
        on_conflict="slug",
    ).execute()
    return "updated"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("dataset", nargs="?", choices=sorted(SOURCES))
    group.add_argument("--all", action="store_true", help="ingest every source")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = get_client()
    for key in sorted(SOURCES) if args.all else [args.dataset]:
        log.info("%s: %s", key, ingest(client, SOURCES[key]))
    check_budget(client)


if __name__ == "__main__":
    main()
