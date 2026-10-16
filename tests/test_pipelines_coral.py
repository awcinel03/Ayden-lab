"""CRW ERDDAP ingester and coral forecast experiment, with Supabase and ERDDAP mocked."""

import gzip
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pipelines.experiments.machine_learning import coral_forecast as cf
from pipelines.ingest import crw_erddap as crw

AXIS_LAT = "latitude\ndegrees_north\n89.975\n89.925\n"
AXIS_LON = "longitude\ndegrees_east\n-179.975\n-179.925\n"


def erddap_chunk(start: str) -> str:
    """Two cells, one land row (all NaN), with the units header row."""
    head = "time,latitude,longitude,CRW_SST,CRW_SSTANOMALY,CRW_DHW\nUTC,deg,deg,C,C,C-week\n"
    return head + (
        f"{start}T12:00:00Z,-10.025,142.025,29.5,0.5,0.1\n"
        f"{start}T12:00:00Z,-10.025,143.025,NaN,NaN,NaN\n"
        f"{start}T12:00:00Z,-11.025,142.025,29.0,0.2,0.0\n"
    )


def fake_fetch(url: str) -> str:
    if "latitude[0:1:1]" in url:
        return AXIS_LAT
    if "longitude[0:1:1]" in url:
        return AXIS_LON
    start = url.split("[(")[1][:10]
    return erddap_chunk(start)


def test_url_orients_descending_latitude():
    url = crw.griddap_url(crw.GBR, "2005-01-01", "2005-03-31", 20,
                          lat_ascending=False, lon_ascending=True)  # fmt: skip
    assert "[(-10.0):20:(-24.5)]" in url and "[(142.0):20:(154.0)]" in url
    assert url.count("[(2005-01-01T12:00:00Z):1:(2005-03-31T12:00:00Z)]") == 3


def test_quarter_chunks_cover_each_year():
    chunks = list(crw.quarter_chunks(2005, 2006))
    assert len(chunks) == 8 and chunks[0] == ("2005-01-01", "2005-03-31")
    assert chunks[-1] == ("2006-10-01", "2006-12-31")


def test_fetch_pull_tidies_drops_land_and_caches(tmp_path):
    pull = crw.Pull("x", crw.GBR, 2005, 2005, 20)
    frame = crw.fetch_pull(pull, fetch=fake_fetch, cache_dir=tmp_path)
    assert len(frame) == 8 and list(frame.columns) == [
        "date", "latitude", "longitude", "CRW_SST", "CRW_SSTANOMALY", "CRW_DHW"
    ]  # fmt: skip
    assert frame["date"].iloc[0] == pd.Timestamp("2005-01-01")
    assert len(list(tmp_path.rglob("*.csv"))) == 4
    offline = crw.fetch_pull(pull, fetch=lambda u: fake_fetch(u) if "[0:1:1]" in u else 1 / 0,
                             cache_dir=tmp_path)  # fmt: skip
    pd.testing.assert_frame_equal(frame, offline)


def test_get_text_retries_reload_404(monkeypatch):
    responses = iter(
        [
            MagicMock(status_code=404, text='message="Currently unknown datasetID=dhw_5km"'),
            MagicMock(status_code=200, text="ok"),
        ]
    )
    monkeypatch.setattr(crw.httpx, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(crw.time, "sleep", lambda s: None)
    assert crw.get_text("u") == "ok"
    monkeypatch.setattr(crw.httpx, "get", lambda *a, **k: MagicMock(status_code=404, text="bad"))
    with pytest.raises(crw.ERDDAPError):
        crw.get_text("u")


def test_ingest_records_sha_of_uncompressed_csv_and_skips_unchanged():
    frame = crw.fetch_pull(crw.Pull("x", crw.GBR, 2005, 2005, 20), fetch=fake_fetch,
                           cache_dir=None)  # fmt: skip
    client = MagicMock()
    datasets = client.table.return_value
    datasets.select.return_value.eq.return_value.execute.return_value.data = []
    assert crw.ingest(client, crw.PULL, frame) == "updated"
    path, blob = client.storage.from_.return_value.upload.call_args.args[:2]
    row = datasets.upsert.call_args.args[0]
    assert path == crw.PULL.path and row["sha256"] == crw.sha256_bytes(gzip.decompress(blob))
    assert row["row_count"] == 8 and "Skirving" in row["citation"]
    datasets.select.return_value.eq.return_value.execute.return_value.data = [
        {"sha256": row["sha256"]}
    ]
    assert crw.ingest(client, crw.PULL, frame) == "unchanged"


def panel(days: int = 60, gap: str | None = None) -> pd.DataFrame:
    dates = pd.date_range("2019-11-01", periods=days)
    frames = []
    for lat in (-10.0, -11.0):
        f = pd.DataFrame({"date": dates, "latitude": lat, "longitude": 142.0})
        f["CRW_SST"] = 28 + np.arange(days) * 0.01
        f["CRW_SSTANOMALY"] = np.arange(days) * 0.01
        f["CRW_DHW"] = np.arange(days, dtype=float)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    return out[out["date"] != pd.Timestamp(gap)] if gap else out


def test_lags_are_date_aware_across_gaps():
    feats = cf.build_features(panel(gap="2019-11-20"), horizon=7)
    row = feats[(feats["latitude"] == -10.0) & (feats["date"] == pd.Timestamp("2019-12-10"))]
    assert row["dhw_lag7"].item() == 32.0 and row["dhw_lag28"].item() == 11.0
    assert row[cf.TARGET].item() == 46.0 and row["dhw_accel7"].item() == 7.0
    # any row whose lag (or the day itself) lands on the missing day is dropped, not mis-lagged
    gap = pd.Timestamp("2019-11-20")
    hits = {gap + pd.Timedelta(days=d) for d in (0, 7, 14, 28)}
    assert not feats["date"].isin(hits).any()


def test_split_embargoes_targets_inside_the_test_window():
    feats = cf.build_features(panel(days=90), horizon=7)
    train, test, embargoed = cf.chronological_split(feats, [2019, 2019], [2020, 2020])
    assert train["target_date"].max() < pd.Timestamp("2020-01-01")
    assert embargoed == 2 * 7 and test["date"].dt.year.eq(2020).all()


def test_timing_finds_lag_and_crossings():
    idx = pd.date_range("2020-01-01", "2020-06-30")
    actual = pd.Series(np.linspace(0, 8, len(idx)), index=idx)
    daily = pd.DataFrame({"actual": actual, "predicted": actual.shift(5).bfill()})
    curve, crossings = cf.timing(daily, threshold=4.0, max_shift=10)
    assert min(curve, key=curve.get) == -5
    assert crossings.iloc[0]["late_days"] == 5


def test_feature_columns_for_each_ablation():
    sets = {k: cf.feature_columns(v["variables"], v["diffs"])
            for k, v in cf.CONFIG["ablation"].items()}  # fmt: skip
    assert [len(v) for v in sets.values()] == [8, 12, 15]
    assert sets["full"][-1] == "dhw_accel7"
