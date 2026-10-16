"""StatsBomb ingester and Arsenal passing-network experiment, with Supabase mocked."""

import gzip
import json
from unittest.mock import MagicMock

from pipelines.experiments.network_science import football as fb
from pipelines.ingest import statsbomb


def ev(player, recipient=None, *, team="Arsenal", kind="Pass", outcome=None, pos="Left Back"):
    """A minimal StatsBomb event; ``recipient`` None means no recipient was recorded."""
    p = {"end_location": [10.0, 20.0]}
    if recipient:
        p["recipient"] = {"name": recipient}
    if outcome:
        p["outcome"] = {"name": outcome}
    return {
        "type": {"name": kind},
        "team": {"name": team},
        "player": {"name": player},
        "position": {"name": pos},
        "location": [0.0, 0.0],
        "pass": p,
    }


EVENTS = [
    ev("A", "B"),
    ev("A", "B"),
    ev("B", "A", pos="Left Midfield"),
    ev("A", "C"),
    ev("C"),  # completed, no recipient: counts as a pass, no edge
    ev("A", "B", outcome="Incomplete"),
    ev("X", "Y", team="Chelsea"),
    ev("A", "B", kind="Carry"),
    ev("A", "B", pos="Left Midfield"),  # A's minority position
]
LINEUPS = [
    {
        "team_name": "Arsenal",
        "lineup": [
            {"player_name": "A", "player_nickname": "Aye"},
            {"player_name": "B", "player_nickname": None},
        ],
    },
    {"team_name": "Chelsea", "lineup": [{"player_name": "X", "player_nickname": "Ex"}]},
]
MATCHES = [
    {
        "match_id": 1,
        "match_date": "2003-10-18",
        "home_team": {"home_team_name": "Arsenal"},
        "away_team": {"away_team_name": "Chelsea"},
        "home_score": 2,
        "away_score": 1,
    }
]
SMALL = fb.CONFIG | {"matches": [1]}


def test_completed_passes_filters_team_type_and_outcome():
    assert len(fb.completed_passes(EVENTS, "Arsenal")) == 6


def test_network_weights_and_skips_missing_recipient():
    g = fb.passing_network(fb.completed_passes(EVENTS, "Arsenal"))
    assert {(u, v): d["weight"] for u, v, d in g.edges(data=True)} == {
        ("A", "B"): 3,
        ("B", "A"): 1,
        ("A", "C"): 1,
    }


def test_average_positions_mix_pass_starts_and_receptions():
    pos = fb.average_positions([ev("A", "B"), ev("B")])
    assert pos["A"] == (0.0, 0.0)
    assert pos["B"] == (5.0, 10.0)  # mean of (0, 0) passing and (10, 20) receiving


def test_main_position_is_the_most_frequent():
    assert fb.main_positions(fb.completed_passes(EVENTS, "Arsenal"))["A"] == "Left Back"


def test_labels_prefer_nickname_for_the_team_only():
    assert fb.labels(LINEUPS, "Arsenal") == {"A": "Aye", "B": "B"}


def test_compute_rows():
    rows = fb.compute(MATCHES, {1: EVENTS}, {1: LINEUPS}, SMALL)
    s = {r["metric"]: r["value"] for r in rows if set(r["dims"]) == {"match"}}
    assert s["completed_passes"] == 6 and s["players"] == 3 and s["connections"] == 3
    assert s["average_degree"] == 2.0
    info = next(r for r in rows if r["metric"] == "match_info")
    assert info["dims"]["home_score"] == 2 and info["dims"]["away"] == "Chelsea"
    made = {r["dims"]["player"]: r for r in rows if r["metric"] == "passes_made"}
    assert made["A"]["value"] == 4 and made["A"]["dims"]["label"] == "Aye"
    assert made["B"]["dims"]["received"] == 3 and made["C"]["value"] == 1
    keys = [(r["metric"], json.dumps(r["dims"], sort_keys=True)) for r in rows]
    assert len(keys) == len(set(keys))


def mock_client(existing_sha=None, exists=True):
    client = MagicMock()
    datasets = MagicMock()
    datasets.select.return_value.eq.return_value.execute.return_value.data = (
        [{"sha256": existing_sha}] if existing_sha else []
    )
    client.table.return_value = datasets
    client.storage.from_.return_value.exists.return_value = exists
    return client, datasets


BLOB = json.dumps(MATCHES).encode()
SOURCE = statsbomb.SOURCES["pl-2003-04-matches"]


def test_ingest_skips_unchanged_file():
    client, datasets = mock_client(existing_sha=statsbomb.sha256_bytes(BLOB))
    assert statsbomb.ingest(client, SOURCE, fetch=lambda url: BLOB) == "unchanged"
    client.storage.from_.return_value.upload.assert_not_called()
    datasets.upsert.assert_not_called()


def test_ingest_records_licence_and_counts_records():
    client, datasets = mock_client()
    assert statsbomb.ingest(client, SOURCE, fetch=lambda url: BLOB) == "updated"
    path, blob = client.storage.from_.return_value.upload.call_args.args[:2]
    assert path == "statsbomb/matches/2/44.json.gz" and gzip.decompress(blob) == BLOB
    row = datasets.upsert.call_args.args[0]
    assert row["row_count"] == 1 and "logo" in row["license"]
    assert row["sha256"] == statsbomb.sha256_bytes(BLOB) and row["bytes"] == len(blob)
    assert SOURCE.url.endswith("/data/matches/2/44.json")


def test_every_configured_dataset_has_a_source():
    assert set(fb.CONFIG["datasets"]) == {s.slug for s in statsbomb.SOURCES.values()}


def test_run_verifies_sha_of_decompressed_json(monkeypatch):
    client = MagicMock()
    raw = json.dumps(MATCHES).encode()
    rows = {
        slug: {"id": i, "sha256": fb.sha256_bytes(raw), "storage_path": f"datasets/{slug}.gz"}
        for i, slug in enumerate(fb.CONFIG["datasets"])
    }
    eq = client.table.return_value.select.return_value.eq
    eq.side_effect = lambda col, slug: MagicMock(**{"execute.return_value.data": [rows[slug]]})
    client.storage.from_.return_value.download.return_value = gzip.compress(raw)
    monkeypatch.setattr(fb, "sync_experiments", lambda c: {fb.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(fb, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(fb, "start_run", lambda c, **kw: 11)
    monkeypatch.setattr(fb, "tracked", lambda c, run_id: MagicMock())
    computed = {}
    monkeypatch.setattr(fb, "compute", lambda m, e, lu, cfg: computed.update(m=m) or [])
    monkeypatch.setattr(fb, "write_results", lambda c, run_id, r: None)
    assert fb.run(client) == 11 and computed["m"] == MATCHES
    client.storage.from_.return_value.download.return_value = gzip.compress(b"[]")
    try:
        fb.run(client)
    except RuntimeError as e:
        assert "sha256" in str(e)
    else:
        raise AssertionError("tampered file accepted")
