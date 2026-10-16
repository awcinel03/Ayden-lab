"""Build the wine flavor co-occurrence aggregates from the local Kaggle CSV.

Run: python -m pipelines.ingest.wine_flavor data/raw/winemag-data-130k-v2.csv

Rights: the review text is Wine Enthusiast's content. The raw CSV is never
uploaded; its ``datasets`` row records the Kaggle URL, stated licence and
sha256 with no storage path. Only derived aggregates (term review counts and
the co-occurrence edge list, no text) go to the private ``datasets`` bucket.
Idempotent: unchanged aggregates with objects present are not rewritten.

Matching is a faithful port of the original notebook: ``term in
description.lower()`` (substring, no word boundaries), reviews with fewer than
two matched terms are skipped, and an edge's weight is the number of reviews
containing both terms (kept if at least ``MIN_EDGE_WEIGHT``).
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from pipelines.common.client import get_client
from pipelines.common.provenance import sha256_bytes
from pipelines.common.storage import check_budget, upload
from supabase import Client

log = logging.getLogger(__name__)

BUCKET = "datasets"
LEXICON_PATH = Path(__file__).with_name("wine_flavor_lexicon_v1.json")
MIN_EDGE_WEIGHT = 10
RAW_SLUG = "kaggle-wine-reviews-130k"
TERMS_SLUG = "wine-flavor-terms-v1"
EDGES_SLUG = "wine-flavor-cooccurrence-v1"
KAGGLE_URL = "https://www.kaggle.com/datasets/zynicide/wine-reviews"
LICENSE = "CC BY-NC-SA 4.0 (as stated on Kaggle; review text © Wine Enthusiast)"
CITATION = "zackthoutt. Wine Reviews (winemag-data-130k-v2.csv). Kaggle, 2017. " + KAGGLE_URL


def load_lexicon(path: Path = LEXICON_PATH) -> tuple[int, list[str]]:
    """Return the lexicon version and its terms in file order."""
    doc = json.loads(path.read_text())
    return doc["version"], [t["term"] for t in doc["terms"]]


@dataclass(frozen=True)
class Aggregates:
    """Term review counts and pair co-occurrence counts, in first-appearance order."""

    term_counts: Counter[str]
    edge_counts: Counter[tuple[str, str]]


def aggregate(descriptions: Iterable[object], lexicon: list[str]) -> Aggregates:
    """Count terms and co-occurring pairs exactly as the notebook does.

    Missing descriptions and reviews with fewer than two matched terms are
    skipped. Within a review, terms are visited in lexicon order, which fixes
    the first-appearance order of nodes (the notebook's set order was not stable).
    """
    term_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    for text in descriptions:
        if pd.isna(text):
            continue
        description = str(text).lower()
        found = [w for w in lexicon if w in description]
        if len(found) < 2:
            continue
        for w in found:
            term_counts[w] += 1
        for pair in combinations(sorted(found), 2):
            edge_counts[pair] += 1
    return Aggregates(term_counts, edge_counts)


def terms_table(agg: Aggregates) -> pd.DataFrame:
    """One row per matched term: ``term``, ``reviews``."""
    return pd.DataFrame({"term": list(agg.term_counts), "reviews": list(agg.term_counts.values())})


def edges_table(agg: Aggregates, min_weight: int = MIN_EDGE_WEIGHT) -> pd.DataFrame:
    """Pairs with weight at least ``min_weight``: ``term_a``, ``term_b``, ``weight``."""
    kept = [(a, b, w) for (a, b), w in agg.edge_counts.items() if w >= min_weight]
    return pd.DataFrame(kept, columns=["term_a", "term_b", "weight"])


def parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a frame to zstd Parquet without the index."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _derived_unchanged(client: Client, slug: str, sha: str, path: str) -> bool:
    rows = client.table("datasets").select("sha256").eq("slug", slug).execute().data
    return bool(rows) and rows[0]["sha256"] == sha and client.storage.from_(BUCKET).exists(path)


def ingest(client: Client, csv_path: Path) -> str:
    """Record the raw CSV, build aggregates, upload them; ``"updated"`` or ``"unchanged"``."""
    raw = csv_path.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), index_col=0)
    version, lexicon = load_lexicon()
    agg = aggregate(df["description"], lexicon)
    raw_sha = sha256_bytes(raw)
    provenance = f"Derived from {RAW_SLUG} (sha256 {raw_sha}) with lexicon v{version}."
    derived = {
        TERMS_SLUG: (
            "Wine flavor term review counts (v1)",
            terms_table(agg),
            "Reviews (with at least two lexicon terms) containing each term. ",
        ),
        EDGES_SLUG: (
            "Wine flavor co-occurrence edge list (v1)",
            edges_table(agg),
            f"Term pairs and the number of reviews containing both; weight >= {MIN_EDGE_WEIGHT}. ",
        ),
    }
    blobs = {slug: parquet_bytes(table) for slug, (_, table, _) in derived.items()}
    paths = {slug: f"derived/{slug}.parquet" for slug in derived}
    unchanged = all(
        _derived_unchanged(client, slug, sha256_bytes(blobs[slug]), paths[slug]) for slug in derived
    )
    raw_row = client.table("datasets").select("sha256").eq("slug", RAW_SLUG).execute().data
    if unchanged and raw_row and raw_row[0]["sha256"] == raw_sha:
        return "unchanged"
    retrieved = _now()
    rows = [
        {
            "slug": RAW_SLUG,
            "name": "Wine Enthusiast reviews (Kaggle winemag-data-130k-v2)",
            "source": "Kaggle (zynicide/wine-reviews); reviews scraped from Wine Enthusiast",
            "source_url": KAGGLE_URL,
            "license": LICENSE,
            "citation": CITATION,
            "storage_path": None,  # never uploaded: the review text is not ours to re-host
            "format": "csv",
            "bytes": len(raw),
            "sha256": raw_sha,
            "row_count": len(df),
            "retrieved_at": retrieved,
            "notes": "Kept local only, placed manually at data/raw/; download not automated.",
        }
    ]
    for slug, (name, table, notes) in derived.items():
        upload(client, BUCKET, paths[slug], blobs[slug], "application/vnd.apache.parquet")
        rows.append(
            {
                "slug": slug,
                "name": name,
                "source": "Derived in this lab from the Kaggle wine reviews",
                "source_url": KAGGLE_URL,
                "license": LICENSE,
                "citation": CITATION,
                "storage_path": f"{BUCKET}/{paths[slug]}",
                "format": "parquet",
                "bytes": len(blobs[slug]),
                "sha256": sha256_bytes(blobs[slug]),
                "row_count": len(table),
                "retrieved_at": retrieved,
                "notes": notes + provenance + " Aggregates only; contains no review text.",
            }
        )
    client.table("datasets").upsert(rows, on_conflict="slug").execute()
    return "updated"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="local path to winemag-data-130k-v2.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = get_client()
    log.info("wine flavor aggregates: %s", ingest(client, args.csv))
    check_budget(client)


if __name__ == "__main__":
    main()
