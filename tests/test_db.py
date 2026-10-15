"""lab.db and lab.storage with Supabase mocked."""

import io
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lab import db, storage


def test_no_secrets_degrades_to_empty():  # conftest hides any local secrets
    assert db.get_client() is None
    assert db.latest_run("x") is None
    assert db.results(1).empty
    assert list(db.results(1).columns) == ["metric", "value", "unit", "dims"]
    assert db.run_datasets(1) == []
    assert db.artifacts(1) == []


def test_latest_run_parses_row(monkeypatch):
    client = MagicMock()
    chain = client.table.return_value.select.return_value.eq.return_value.limit.return_value
    chain.execute.return_value.data = [
        {
            "id": 7,
            "experiment_slug": "graph-stats",
            "started_at": "2026-10-15T12:00:00+00:00",
            "finished_at": None,
            "config": {"k": 1},
            "git_sha": "abc",
        }
    ]
    monkeypatch.setattr(db, "get_client", lambda: client)
    run = db.latest_run("graph-stats")
    assert run.id == 7 and run.finished_at is None and run.started_at.year == 2026
    client.table.assert_called_with("latest_runs")


def test_results_pages_past_max_rows(monkeypatch):
    monkeypatch.setattr(db, "PAGE_SIZE", 2)
    pages = [
        [{"metric": "a", "value": 1, "unit": None, "dims": {}}] * 2,
        [{"metric": "b", "value": 2, "unit": None, "dims": {}}],
    ]
    client = MagicMock()
    ranged = client.table.return_value.select.return_value.eq.return_value.order.return_value.range
    ranged.return_value.execute.side_effect = [MagicMock(data=p) for p in pages]
    monkeypatch.setattr(db, "get_client", lambda: client)
    df = db.results(1)
    assert len(df) == 3
    assert [c.args for c in ranged.call_args_list] == [(0, 1), (2, 3)]


def test_api_error_degrades_to_empty(monkeypatch):
    client = MagicMock()
    client.table.side_effect = RuntimeError("network down")
    monkeypatch.setattr(db, "get_client", lambda: client)
    assert db.results(1).empty
    assert db.latest_run("x") is None


def artifact(nbytes):
    return db.Artifact(kind="table", storage_path="s/1/t.parquet", bytes=nbytes, content_type=None)


def test_oversized_artifact_is_not_downloaded(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(db, "get_client", lambda: client)
    assert storage.artifact_bytes(artifact(storage.MAX_APP_BYTES + 1)) is None
    client.storage.from_.assert_not_called()


def test_parquet_artifact_round_trip(monkeypatch):
    buf = io.BytesIO()
    pd.DataFrame({"x": [1, 2]}).to_parquet(buf)
    client = MagicMock()
    client.storage.from_.return_value.download.return_value = buf.getvalue()
    monkeypatch.setattr(db, "get_client", lambda: client)
    assert storage.read_parquet(artifact(10)).x.tolist() == [1, 2]


def test_download_error_returns_none(monkeypatch):
    client = MagicMock()
    client.storage.from_.return_value.download.side_effect = RuntimeError("403")
    monkeypatch.setattr(db, "get_client", lambda: client)
    assert storage.read_parquet(artifact(10)) is None


@pytest.mark.parametrize(
    ("key", "ok"),
    [
        ("sb_publishable_" + "Ab-1_", True),  # split so secret scanners skip it
        ("sb_publishable_…", False),  # the docs' old placeholder
        ("<paste the full publishable key>", False),
        ("sb_secret_" + "Ab-1_", False),
        ("sb_publishable_", False),
        ("", False),
        (None, False),
    ],
)
def test_publishable_key_format(key, ok):
    assert db.publishable_key_looks_valid(key) is ok


def test_placeholder_key_gives_no_client(monkeypatch):
    monkeypatch.setattr(
        db.st,
        "secrets",
        {"SUPABASE_URL": "https://x.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "sb_publishable_…"},
    )
    db.get_client.clear()
    try:
        assert db.get_client() is None
    finally:
        db.get_client.clear()
