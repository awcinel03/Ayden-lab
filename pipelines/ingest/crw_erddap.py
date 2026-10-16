"""Ingest NOAA Coral Reef Watch SST, SSTA and DHW for the Great Barrier Reef from ERDDAP.

Run: python -m pipelines.ingest.crw_erddap [--refresh]

A port of the ERDDAP client in my coral_bleaching project. Requests the daily
5 km CRW v3.1 grid one calendar quarter at a time (a 19-year request in one go
times out), every 20th cell in each direction (~1°), 2005-2023. Each raw
response is cached under ``data/raw/crw/`` (gitignored) so an interrupted pull
resumes; ``--refresh`` ignores the cache. The tidy rows are combined into one
gzipped CSV in the private ``datasets`` bucket.

Idempotent: if the combined CSV's sha256 (of the uncompressed bytes) matches the
recorded row and the object exists, nothing is written. The server is slow and
uneven (~1.5 h for a cold pull); requests are sequential with retries.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import io
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from pipelines.common.client import get_client
from pipelines.common.provenance import REPO_ROOT, sha256_bytes
from pipelines.common.storage import check_budget, upload
from pipelines.ingest.snap import USER_AGENT
from supabase import Client

log = logging.getLogger(__name__)

BUCKET = "datasets"
SERVER = "https://pae-paha.pacioos.hawaii.edu/erddap"
DATASET_ID = "dhw_5km"
VARIABLES = ("CRW_SST", "CRW_SSTANOMALY", "CRW_DHW")
CACHE_DIR = REPO_ROOT / "data" / "raw" / "crw"
LICENSE = (
    "NOAA Coral Reef Watch: available for use without restriction, with credit to NOAA CRW; "
    "GHRSST input (2002-present) free and open. (The dataset's OSTIA terms cover 1985-2002 "
    "only, outside this pull.)"
)
CITATION = (
    "NOAA Coral Reef Watch. 2018, updated daily. NOAA Coral Reef Watch Version 3.1 Daily Global "
    "5km Satellite Coral Bleaching Degree Heating Week Product, 2005-01-01 to 2023-12-31. "
    "College Park, Maryland, USA: NOAA Coral Reef Watch. Accessed via PacIOOS ERDDAP "
    "(dhw_5km). | Skirving, W. et al. CoralTemp and the Coral Reef Watch Coral Bleaching Heat "
    "Stress Product Suite Version 3.1. Remote Sens. 12, 3856 (2020). "
    "https://doi.org/10.3390/rs12233856"
)


@dataclass(frozen=True)
class Region:
    """A rectangular region in degrees (longitude -180..180)."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


GBR = Region("great_barrier_reef", -24.5, -10.0, 142.0, 154.0)


@dataclass(frozen=True)
class Pull:
    """What to request: region, years and grid stride."""

    slug: str
    region: Region
    start_year: int
    end_year: int
    stride: int

    @property
    def path(self) -> str:
        """Object path inside the ``datasets`` bucket."""
        r = self.region.name
        return f"crw/{r}_{self.start_year}_{self.end_year}_stride{self.stride}.csv.gz"


PULL = Pull("crw-dhw-gbr-2005-2023-stride20", GBR, 2005, 2023, 20)


class ERDDAPError(RuntimeError):
    """ERDDAP returned an error or stayed unreachable after retries."""


def get_text(url: str, retries: int = 4, backoff: float = 3.0, timeout: float = 300) -> str:
    """GET ``url`` with linear backoff; 400/403/404 are query errors and are not retried,
    except the 404 ERDDAP returns while a dataset is reloading."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if response.status_code == 200:
                return response.text
            message = f"HTTP {response.status_code} from ERDDAP: {response.text[:500]}"
            # "Currently unknown datasetID" is a 404 ERDDAP sends while reloading a dataset.
            reloading = "Currently unknown datasetID" in response.text
            if response.status_code in (400, 403, 404) and not reloading:
                raise ERDDAPError(message)
            last = ERDDAPError(message)
        except httpx.HTTPError as exc:
            last = ERDDAPError(f"Network error contacting ERDDAP: {exc}")
        if attempt < retries:
            time.sleep(backoff * attempt)
    raise last or ERDDAPError("Unknown ERDDAP failure")


def read_erddap_csv(text: str) -> pd.DataFrame:
    """Parse an ERDDAP CSV, skipping its second header row (units)."""
    return pd.read_csv(io.StringIO(text), skiprows=[1])


def axis_is_ascending(fetch: Callable[[str], str], axis: str) -> bool:
    """Whether ``axis`` increases with index (CRW latitude runs north to south)."""
    url = f"{SERVER}/griddap/{DATASET_ID}.csv?{axis}[0:1:1]"
    values = read_erddap_csv(fetch(url))[axis].astype(float).tolist()
    if len(values) < 2:
        raise ERDDAPError(f"Could not read the {axis!r} axis")
    return values[1] > values[0]


def griddap_url(
    region: Region,
    start: str,
    end: str,
    stride: int,
    *,
    lat_ascending: bool,
    lon_ascending: bool,
    variables: Sequence[str] = VARIABLES,
) -> str:
    """One request for every variable over a region and date span (CRW stamps 12:00Z)."""
    lat = (region.lat_min, region.lat_max) if lat_ascending else (region.lat_max, region.lat_min)
    lon = (region.lon_min, region.lon_max) if lon_ascending else (region.lon_max, region.lon_min)
    subset = (
        f"[({start}T12:00:00Z):1:({end}T12:00:00Z)]"
        f"[({lat[0]}):{stride}:({lat[1]})]"
        f"[({lon[0]}):{stride}:({lon[1]})]"
    )
    return f"{SERVER}/griddap/{DATASET_ID}.csv?" + ",".join(f"{v}{subset}" for v in variables)


def quarter_chunks(start_year: int, end_year: int) -> Iterator[tuple[str, str]]:
    """``(start, end)`` dates of every calendar quarter in the span."""
    spans = (("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31"))
    for year in range(start_year, end_year + 1):
        for a, b in spans:
            yield f"{year}-{a}", f"{year}-{b}"


def tidy(frame: pd.DataFrame, variables: Sequence[str] = VARIABLES) -> pd.DataFrame:
    """``date``, coordinates rounded to 4 dp, numeric variables; all-NaN (land) rows dropped."""
    out = frame.copy()
    out["date"] = (
        pd.to_datetime(out["time"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    )
    out["latitude"] = out["latitude"].astype(float).round(4)
    out["longitude"] = out["longitude"].astype(float).round(4)
    for v in variables:
        out[v] = pd.to_numeric(out[v], errors="coerce")
    out = out.dropna(subset=list(variables), how="all")
    return out[["date", "latitude", "longitude", *variables]].reset_index(drop=True)


def fetch_pull(
    pull: Pull,
    fetch: Callable[[str], str] = get_text,
    cache_dir: Path | None = CACHE_DIR,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Every quarter of ``pull``, tidied and concatenated.

    Raw responses are written to ``cache_dir`` (if given) and reused on later runs
    unless ``refresh`` is set.
    """
    lat_up = axis_is_ascending(fetch, "latitude")
    lon_up = axis_is_ascending(fetch, "longitude")
    chunks = list(quarter_chunks(pull.start_year, pull.end_year))
    frames = []
    for i, (start, end) in enumerate(chunks, start=1):
        cached = (
            cache_dir / pull.region.name / f"{start}_{end}_stride{pull.stride}.csv"
            if cache_dir
            else None
        )
        if cached and not refresh and cached.is_file() and cached.stat().st_size:
            text, source = cached.read_text(), "cache"
        else:
            url = griddap_url(
                pull.region, start, end, pull.stride, lat_ascending=lat_up, lon_ascending=lon_up
            )
            t0 = time.monotonic()
            text, source = fetch(url), f"download {time.monotonic() - t0:.0f}s"
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(text)
        frames.append(tidy(read_erddap_csv(text)))
        log.info(
            "[%d/%d] %s -> %s: %d rows (%s)", i, len(chunks), start, end, len(frames[-1]), source
        )
    return pd.concat(frames, ignore_index=True)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Deterministic CSV encoding of the tidy table (dates as YYYY-MM-DD)."""
    return frame.to_csv(index=False, date_format="%Y-%m-%d").encode()


def ingest(client: Client, pull: Pull, frame: pd.DataFrame) -> str:
    """Upload the combined table and upsert its row; ``"unchanged"`` if already recorded."""
    data = to_csv_bytes(frame)
    sha = sha256_bytes(data)
    rows = client.table("datasets").select("sha256").eq("slug", pull.slug).execute().data
    if rows and rows[0]["sha256"] == sha and client.storage.from_(BUCKET).exists(pull.path):
        return "unchanged"
    blob = gzip.compress(data, mtime=0)
    upload(client, BUCKET, pull.path, blob, "application/gzip")
    r = pull.region
    client.table("datasets").upsert(
        {
            "slug": pull.slug,
            "name": "NOAA CRW SST, SST anomaly and DHW: Great Barrier Reef, 2005-2023",
            "source": "NOAA Coral Reef Watch via PacIOOS ERDDAP",
            "source_url": f"{SERVER}/griddap/{DATASET_ID}.html",
            "license": LICENSE,
            "citation": CITATION,
            "storage_path": f"{BUCKET}/{pull.path}",
            "format": "csv.gz",
            "bytes": len(blob),
            "sha256": sha,  # of the uncompressed CSV
            "row_count": len(frame),
            "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "notes": (
                f"Daily {', '.join(VARIABLES)} for lat {r.lat_min}..{r.lat_max}, lon "
                f"{r.lon_min}..{r.lon_max}, every {pull.stride}th 0.05° cell, "
                f"{pull.start_year}-{pull.end_year}, requested one quarter at a time. Columns: "
                "date, latitude, longitude, then the variables; land/missing pixels dropped."
            ),
        },
        on_conflict="slug",
    ).execute()
    return "updated"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download every chunk")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    frame = fetch_pull(PULL, refresh=args.refresh)
    client = get_client()
    log.info("%s: %s (%d rows)", PULL.slug, ingest(client, PULL, frame), len(frame))
    check_budget(client)


if __name__ == "__main__":
    main()
