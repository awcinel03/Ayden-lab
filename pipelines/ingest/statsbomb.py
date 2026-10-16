"""Ingest StatsBomb open-data JSON files: download → checksum → upload to Storage → upsert row.

Run: python -m pipelines.ingest.statsbomb pl-2003-04-matches
     python -m pipelines.ingest.statsbomb --all
     python -m pipelines.ingest.statsbomb --season   # every match's events and lineups

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
import io
import json
import logging
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipelines.common.client import get_client
from pipelines.common.provenance import REPO_ROOT, sha256_bytes
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
CACHE_DIR = REPO_ROOT / "data" / "raw" / "statsbomb"  # gitignored; avoids re-fetching on reruns
SEASON_MATCHES = "pl-2003-04-matches"


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


def season_slug(kind: str) -> str:
    """Dataset slug of the season bundle for ``kind`` ("events" or "lineups")."""
    return f"statsbomb-pl-2003-04-{kind}"


def fetch_cached(
    path: str, fetch: Callable[[str], bytes] = download, cache_dir: Path | None = CACHE_DIR
) -> bytes:
    """Bytes of ``data/<path>``, from the local cache when present, else downloaded and cached."""
    cached = cache_dir / path if cache_dir else None
    if cached and cached.is_file() and cached.stat().st_size:
        return cached.read_bytes()
    data = fetch(f"{BASE_URL}/{path}")
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    return data


def tar_bundle(files: dict[str, bytes]) -> bytes:
    """Deterministic tar of ``{name: bytes}``: sorted names, zero mtime/owner, bytes as given."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    return buf.getvalue()


def read_bundle(tar_bytes: bytes) -> dict[str, bytes]:
    """Inverse of :func:`tar_bundle`."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def ingest_season(
    client: Client,
    kind: str,
    match_ids: list[int],
    fetch: Callable[[str], bytes] = download,
    cache_dir: Path | None = CACHE_DIR,
) -> str:
    """Bundle every match's ``kind`` file into one gzipped tar and record one dataset row.

    The recorded sha256 is of the uncompressed tar, which holds StatsBomb's files
    byte for byte. Returns ``"unchanged"`` or ``"updated"``.
    """
    files = {
        f"{kind}/{m}.json": fetch_cached(f"{kind}/{m}.json", fetch, cache_dir) for m in match_ids
    }
    data = tar_bundle(files)
    sha, slug = sha256_bytes(data), season_slug(kind)
    path = f"statsbomb/{slug}.tar.gz"
    rows = client.table("datasets").select("sha256").eq("slug", slug).execute().data
    if rows and rows[0]["sha256"] == sha and client.storage.from_(BUCKET).exists(path):
        return "unchanged"
    blob = gzip.compress(data, mtime=0)
    upload(client, BUCKET, path, blob, "application/gzip")
    client.table("datasets").upsert(
        {
            "slug": slug,
            "name": f"StatsBomb {kind}: Arsenal, Premier League 2003/04 ({len(match_ids)} matches)",
            "source": "StatsBomb Open Data",
            "source_url": SOURCE_PAGE,
            "license": LICENSE,
            "citation": CITATION,
            "storage_path": f"{BUCKET}/{path}",
            "format": "tar.gz of json",
            "bytes": len(blob),
            "sha256": sha,  # of the uncompressed tar
            "row_count": sum(count_records(b) for b in files.values()),
            "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "notes": (
                f"Tar of {BASE_URL}/{kind}/<match_id>.json for every match in matches/2/44.json, "
                "original bytes, sorted by name. row_count sums the records in every file."
            ),
        },
        on_conflict="slug",
    ).execute()
    return "updated"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("dataset", nargs="?", choices=sorted(SOURCES))
    group.add_argument("--all", action="store_true", help="ingest every single-file source")
    group.add_argument("--season", action="store_true", help="matches + season bundles")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    client = get_client()
    if args.season:
        log.info("%s: %s", SEASON_MATCHES, ingest(client, SOURCES[SEASON_MATCHES]))
        matches = json.loads(fetch_cached("matches/2/44.json"))
        ids = sorted(m["match_id"] for m in matches)
        for kind in ("events", "lineups"):
            log.info("%s: %s", season_slug(kind), ingest_season(client, kind, ids))
    else:
        for key in sorted(SOURCES) if args.all else [args.dataset]:
            log.info("%s: %s", key, ingest(client, SOURCES[key]))
    check_budget(client)


if __name__ == "__main__":
    main()
