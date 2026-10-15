"""pipelines.common with Supabase mocked."""

import datetime
from unittest.mock import MagicMock

import pytest

from lab.registry import Category, Entry, EntryType
from pipelines.common import client as client_mod
from pipelines.common import provenance, runs, storage, sync


def tables(**by_name):
    """A mock client whose ``table(name)`` returns a dedicated mock per table."""
    client = MagicMock()
    mocks = {
        name: by_name.get(name, MagicMock())
        for name in ("runs", "results", "artifacts", "run_datasets", "experiments", "datasets")
    }
    client.table.side_effect = lambda name: mocks[name]
    return client, mocks


# provenance


def test_sha256_file_matches_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert provenance.sha256_file(p) == provenance.sha256_bytes(b"hello")


def test_canonical_json_is_order_independent():
    assert provenance.canonical_json({"b": 1, "a": 2}) == provenance.canonical_json(
        {"a": 2, "b": 1}
    )


def test_git_sha_refuses_dirty_tree(monkeypatch):
    monkeypatch.setattr(
        provenance, "_git", lambda *a: " M lab/db.py" if a[0] == "status" else "sha"
    )
    with pytest.raises(RuntimeError, match="uncommitted"):
        provenance.git_sha()
    assert provenance.git_sha(allow_dirty=True) == "sha"


# client


def test_client_requires_env(monkeypatch):
    monkeypatch.setattr(client_mod, "load_dotenv", lambda: None)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Set SUPABASE_URL"):
        client_mod.get_client()


def test_client_rejects_publishable_key(monkeypatch):
    monkeypatch.setattr(client_mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_publishable_abc")
    with pytest.raises(RuntimeError, match="publishable"):
        client_mod.get_client()


# runs


def test_fingerprint_is_stable_and_sensitive():
    fp = runs.fingerprint({"a": 1}, "sha", ["d2", "d1"])
    assert fp == runs.fingerprint({"a": 1}, "sha", ["d1", "d2"])
    assert fp != runs.fingerprint({"a": 2}, "sha", ["d1", "d2"])
    assert fp != runs.fingerprint({"a": 1}, "other", ["d1", "d2"])


def test_start_run_skips_identical_succeeded_run():
    client, t = tables()
    t["runs"].select.return_value.eq.return_value.execute.return_value.data = [
        {"id": 1, "status": "succeeded"}
    ]
    assert runs.start_run(client, experiment_id=1, config={}, git_sha="s", datasets={}) is None
    t["runs"].insert.assert_not_called()


def test_start_run_inserts_and_links_datasets():
    client, t = tables()
    t["runs"].select.return_value.eq.return_value.execute.return_value.data = []
    t["runs"].insert.return_value.execute.return_value.data = [{"id": 9}]
    run_id = runs.start_run(client, experiment_id=1, config={}, git_sha="s", datasets={4: "h"})
    assert run_id == 9
    rows = t["run_datasets"].upsert.call_args.args[0]
    assert rows == [{"run_id": 9, "dataset_id": 4}]


def test_start_run_reuses_failed_run_and_clears_outputs():
    client, t = tables()
    t["runs"].select.return_value.eq.return_value.execute.return_value.data = [
        {"id": 3, "status": "failed"}
    ]
    assert runs.start_run(client, experiment_id=1, config={}, git_sha="s", datasets={}) == 3
    t["results"].delete.return_value.eq.assert_called_with("run_id", 3)
    t["artifacts"].delete.return_value.eq.assert_called_with("run_id", 3)
    t["runs"].insert.assert_not_called()


def test_tracked_marks_failure_and_reraises():
    client, t = tables()
    with pytest.raises(ValueError), runs.tracked(client, 5):
        raise ValueError("boom")
    assert t["runs"].update.call_args.args[0]["status"] == "failed"


def test_tracked_marks_success():
    client, t = tables()
    with runs.tracked(client, 5):
        pass
    assert t["runs"].update.call_args.args[0]["status"] == "succeeded"


def test_write_results_batches_and_defaults(monkeypatch):
    monkeypatch.setattr(runs, "RESULTS_BATCH", 2)
    client, t = tables()
    runs.write_results(client, 1, [{"metric": "m", "value": i} for i in range(5)])
    batches = [c.args[0] for c in t["results"].upsert.call_args_list]
    assert [len(b) for b in batches] == [2, 2, 1]
    assert batches[0][0] == {"run_id": 1, "metric": "m", "value": 0, "unit": None, "dims": {}}


# sync


def test_experiment_row_uses_category_key():
    entry = Entry(
        slug="graph-stats",
        title="Graph stats",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.EXPERIMENT,
        path="app_pages/network_science/graph_stats.py",
        summary="Stats.",
        date=datetime.date(2026, 10, 15),
    )
    assert sync.experiment_row(entry)["category"] == "network_science"
    client, t = tables()
    t["experiments"].upsert.return_value.execute.return_value.data = [
        {"slug": "graph-stats", "id": 2}
    ]
    assert sync.sync_experiments(client, (entry,)) == {"graph-stats": 2}
    assert t["experiments"].upsert.call_args.kwargs["on_conflict"] == "slug"


# storage


def test_upload_rejects_files_over_free_limit():
    with pytest.raises(ValueError, match="50 MB"):
        storage.upload(MagicMock(), "datasets", "x", b"0" * (storage.FREE_PLAN_FILE_LIMIT + 1), "x")


def test_check_budget_warns(caplog):
    client, t = tables()
    for name in ("datasets", "artifacts"):
        t[name].select.return_value.execute.return_value.data = [{"bytes": 400 * 2**20}]
    assert storage.check_budget(client) == 800 * 2**20
    assert "over the 750 MB budget" in caplog.text
