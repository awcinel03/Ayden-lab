"""Ingest a SNAP dataset: download → checksum → upload to Storage → upsert ``datasets`` row.

Run: python -m pipelines.ingest.snap email-Eu-core

Idempotent: if the downloaded file's sha256 matches the recorded row and the
object exists in Storage, nothing is written. SNAP states no licence and asks
for citations; one polite request per run with a descriptive User-Agent.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from pipelines.common.client import get_client
from pipelines.common.provenance import sha256_bytes
from pipelines.common.storage import check_budget, upload
from supabase import Client

log = logging.getLogger(__name__)

BUCKET = "datasets"
USER_AGENT = "aydens-lab-ingest/0.1 (+https://github.com/awcinel03/Ayden-lab)"
SNAP_CITATION = (
    "Jure Leskovec and Andrej Krevl. SNAP Datasets: Stanford Large Network Dataset "
    "Collection. http://snap.stanford.edu/data, 2014."
)


@dataclass(frozen=True)
class SnapSource:
    """A SNAP file and its provenance metadata."""

    slug: str
    name: str
    url: str
    filename: str
    citation: str
    notes: str


SOURCES: dict[str, SnapSource] = {
    "email-Eu-core": SnapSource(
        slug="snap-email-eu-core",
        name="email-Eu-core network",
        url="https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        filename="email-Eu-core.txt.gz",
        citation=(
            "Hao Yin, Austin R. Benson, Jure Leskovec, David F. Gleich. Local Higher-order "
            "Graph Clustering. KDD 2017. | Jure Leskovec, Jon Kleinberg, Christos Faloutsos. "
            "Graph Evolution: Densification and Shrinking Diameters. ACM TKDD 1(1), 2007. | "
            + SNAP_CITATION
        ),
        notes=(
            "Directed edge list (one 'u v' pair per line) of anonymised email between members "
            "of a European research institution. Node IDs are integers."
        ),
    ),
}


def download(url: str) -> bytes:
    """Fetch ``url`` with the project User-Agent; raises on HTTP errors."""
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response.content


def count_edges(gz_data: bytes) -> int:
    """Number of non-comment, non-blank lines in a gzipped edge list."""
    lines = gzip.decompress(gz_data).decode().splitlines()
    return sum(1 for line in lines if line.strip() and not line.startswith("#"))


def ingest(client: Client, source: SnapSource, fetch: Callable[[str], bytes] = download) -> str:
    """Mirror ``source`` into Storage and upsert its row.

    Returns ``"unchanged"`` if the file matches the recorded sha256, else ``"updated"``.
    """
    data = fetch(source.url)
    sha = sha256_bytes(data)
    path = f"snap/{source.filename}"
    rows = client.table("datasets").select("sha256").eq("slug", source.slug).execute().data
    if rows and rows[0]["sha256"] == sha and client.storage.from_(BUCKET).exists(path):
        return "unchanged"
    upload(client, BUCKET, path, data, "application/gzip")
    client.table("datasets").upsert(
        {
            "slug": source.slug,
            "name": source.name,
            "source": "Stanford SNAP",
            "source_url": source.url,
            "license": None,  # not stated by SNAP
            "citation": source.citation,
            "storage_path": f"{BUCKET}/{path}",
            "format": "edgelist.txt.gz",
            "bytes": len(data),
            "sha256": sha,
            "row_count": count_edges(data),
            "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "notes": source.notes,
        },
        on_conflict="slug",
    ).execute()
    return "updated"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(SOURCES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = get_client()
    result = ingest(client, SOURCES[args.dataset])
    log.info("%s: %s", args.dataset, result)
    check_budget(client)


if __name__ == "__main__":
    main()
