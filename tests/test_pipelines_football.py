"""StatsBomb ingester and Arsenal passing-network experiment, with Supabase mocked."""

import gzip
import io
import json
import random
import tarfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipelines.experiments.network_science import football as fb
from pipelines.ingest import statsbomb

XI = [
    ("A", "Goalkeeper"),
    ("B", "Left Back"),
    ("C", "Left Center Back"),
    ("D", "Right Center Back"),
]


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


def numbered(events, minute_every=10):
    """Give events StatsBomb's ``index`` and ``minute`` fields, in order."""
    return [e | {"index": i + 1, "minute": i // minute_every} for i, e in enumerate(events)]


def with_possessions(events, spans):
    """Tag events with StatsBomb's ``possession`` and ``possession_team``.

    ``spans`` is a list of (last index in the possession, team), in order.
    """
    out, k = [], 0
    for e in events:
        while e["index"] > spans[k][0]:
            k += 1
        out.append(e | {"possession": k + 1, "possession_team": {"name": spans[k][1]}})
    return out


def starting_xi(team="Arsenal", xi=XI):
    lineup = [{"player": {"name": n}, "position": {"name": p}} for n, p in xi]
    return {"type": {"name": "Starting XI"}, "team": {"name": team}, "tactics": {"lineup": lineup}}


EVENTS = with_possessions(numbered(
    [
        starting_xi(),
        ev("A", "B"),
        ev("A", "B"),
        ev("B", "A", pos="Left Midfield"),
        ev("A", "C"),
        ev("C"),  # completed, no recipient: counts as a pass, no edge
        ev("A", "B", outcome="Incomplete"),
        ev("X", "Y", team="Chelsea"),
        ev("A", "B", kind="Carry"),
        ev("A", "B", pos="Left Midfield"),  # A's minority position
        ev("C", "D"),
        ev("D", "C"),
        {"type": {"name": "Substitution"}, "team": {"name": "Arsenal"}, "player": {"name": "D"}},
        ev("B", "C"),  # after the first substitution: full match only
    ]
), [(7, "Arsenal"), (8, "Chelsea"), (99, "Arsenal")])  # fmt: skip
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
SMALL = fb.CONFIG | {
    "featured_matches": [1],
    "bootstrap": fb.CONFIG["bootstrap"] | {"reps": 20},
    "null": fb.CONFIG["null"] | {"reps": 20},
    "random_partitions": fb.CONFIG["random_partitions"] | {"reps": 20},
    "season_tests": {"permutations": 20, "seed": 0},
}


def test_completed_passes_filters_team_type_and_outcome():
    assert len(fb.completed_passes(EVENTS, "Arsenal")) == 9


def test_network_weights_and_skips_missing_recipient():
    g = fb.passing_network(fb.completed_passes(EVENTS, "Arsenal"))
    assert {(u, v): d["weight"] for u, v, d in g.edges(data=True)} == {
        ("A", "B"): 3,
        ("B", "A"): 1,
        ("A", "C"): 1,
        ("C", "D"): 1,
        ("D", "C"): 1,
        ("B", "C"): 1,
    }


def test_average_positions_mix_pass_starts_and_receptions():
    pos = fb.average_positions([ev("A", "B"), ev("B")])
    assert pos["A"] == (0.0, 0.0)
    assert pos["B"] == (5.0, 10.0)  # mean of (0, 0) passing and (10, 20) receiving


def test_main_position_is_the_most_frequent():
    assert fb.main_positions(fb.completed_passes(EVENTS, "Arsenal"))["A"] == "Left Back"


def test_labels_prefer_nickname_for_the_team_only():
    assert fb.labels(LINEUPS, "Arsenal") == {"A": "Aye", "B": "B"}


def test_roles_and_lines():
    assert [fb.role_of(p) for _, p in XI] == ["GK", "L-DEF", "C-DEF", "C-DEF"]
    assert fb.role_of("Right Midfield") == "R-MID" and fb.role_of("Center Forward") == "C-ATT"
    assert fb.line_of("Left Wing") == "ATT" and fb.line_of("Right Wing Back") == "DEF"
    assert fb.line_of("Center Defensive Midfield") == "MID"


def test_window_stops_at_first_substitution():
    names, positions, counts, minutes = fb.window_counts(EVENTS, "Arsenal")
    assert names == ["A", "B", "C", "D"] and positions[0] == "Goalkeeper"
    assert counts.sum() == 7  # B -> C after the substitution is excluded
    assert counts[0, 1] == 3 and counts[2, 3] == 1 and counts[1, 2] == 0
    assert minutes == 1


def test_window_stops_at_red_card_but_not_yellow():
    yellow = {"type": {"name": "Foul Committed"}, "team": {"name": "Arsenal"},
              "foul_committed": {"card": {"name": "Yellow Card"}}}  # fmt: skip
    red = yellow | {"foul_committed": {"card": {"name": "Second Yellow"}}}
    events = numbered([starting_xi(), ev("A", "B"), yellow, ev("A", "B"), red, ev("A", "B")])
    assert fb.window_counts(events, "Arsenal")[2].sum() == 2


def test_window_stats_on_a_known_matrix():
    counts = np.array([[0, 2, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    s = fb.window_stats(counts, minutes=2, roles=["GK", "L-DEF", "C-DEF", "C-DEF"],
                        lines=["GK", "DEF", "DEF", "DEF"])  # fmt: skip
    assert s["passes"] == 6 and s["intensity"] == 3
    assert s["density"] == 5 / 12
    assert s["reciprocity"] == 4 / 6 and s["top_pair_share"] == 3 / 6  # pairs A-B, C-D
    # involvement s = [3, 4, 3, 2]; sum(4 - s_i) = 1 + 0 + 1 + 2 = 4; (n - 2) P = 12
    assert s["centralization"] == 4 / 12
    assert np.isclose(s["mean_entropy"], 1 / 4)  # only B splits its passes (1 bit)


def test_resample_keeps_pass_total_and_support():
    counts = np.array([[0, 5], [3, 0]])
    out = fb.resample_counts(counts, random.Random(0))
    assert out.sum() == 8 and out[0, 0] == 0 and out[1, 1] == 0


def test_permutation_spearman_detects_monotone_relation():
    x = np.arange(12.0)
    rho, p = fb.permutation_spearman(x, x**2, 200, random.Random(0))
    assert rho == 1.0 and p < 0.05


def test_compute_rows():
    rows = fb.compute(MATCHES, {1: EVENTS}, {1: LINEUPS}, SMALL)
    s = {r["metric"]: r["value"] for r in rows if set(r["dims"]) == {"match"}}
    assert s["completed_passes"] == 9 and s["players"] == 4 and s["connections"] == 6
    assert s["w_passes"] == 7 and s["window_minutes"] == 1
    for stat in fb.WINDOW_STATS:
        assert f"w_{stat}_lo" in s and f"w_{stat}_bias" in s
    for stat in fb.NULL_STATS:
        assert 0 <= s[f"w_{stat}_null_pct"] <= 1
    info = next(r for r in rows if r["metric"] == "match_info")
    assert info["dims"]["result"] == "W" and info["dims"]["opponent"] == "Chelsea"
    made = {r["dims"]["player"]: r for r in rows if r["metric"] == "passes_made"}
    assert made["A"]["value"] == 4 and made["A"]["dims"]["label"] == "Aye"
    window = {r["dims"]["player"]: r for r in rows if r["metric"] == "window_player"}
    assert window["C"]["dims"]["role"] == "C-DEF" and window["A"]["value"] == 4
    assert any(r["metric"] == "role_transition" for r in rows)
    pooled = {r["metric"]: r["value"] for r in rows if not r["dims"]}
    assert pooled["pooled_passes"] == 7 and pooled["season_matches"] == 1
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


def test_tar_bundle_is_deterministic_and_round_trips():
    files = {"events/2.json": b"[2]", "events/1.json": b"[1, 1]"}
    tar = statsbomb.tar_bundle(files)
    assert tar == statsbomb.tar_bundle(dict(reversed(list(files.items()))))
    assert statsbomb.read_bundle(tar) == files
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        assert [m.name for m in t.getmembers()] == ["events/1.json", "events/2.json"]
        assert {m.mtime for m in t.getmembers()} == {0}


def test_ingest_season_bundles_cached_files(tmp_path):
    fetched = []

    def fetch(url):
        fetched.append(url)
        return b"[" + url.rsplit("/", 1)[1].split(".")[0].encode() + b"]"

    client, datasets = mock_client()
    assert statsbomb.ingest_season(client, "lineups", [2, 1], fetch, tmp_path) == "updated"
    path, blob = client.storage.from_.return_value.upload.call_args.args[:2]
    row = datasets.upsert.call_args.args[0]
    assert path == "statsbomb/statsbomb-pl-2003-04-lineups.tar.gz"
    assert row["sha256"] == statsbomb.sha256_bytes(gzip.decompress(blob))
    assert row["row_count"] == 2 and row["slug"] == statsbomb.season_slug("lineups")
    assert len(fetched) == 2
    statsbomb.ingest_season(client, "lineups", [2, 1], fetch, tmp_path)
    assert len(fetched) == 2  # second pass served from the cache


def test_every_configured_dataset_is_produced_by_the_ingester():
    produced = {statsbomb.SOURCES[statsbomb.SEASON_MATCHES].slug} | {
        statsbomb.season_slug(k) for k in ("events", "lineups")
    }
    assert set(fb.CONFIG["datasets"]) == produced


def test_run_verifies_sha_of_every_bundle(monkeypatch):
    client = MagicMock()
    matches = json.dumps(MATCHES).encode()
    bundles = {
        fb.MATCHES_SLUG: matches,
        statsbomb.season_slug("events"): statsbomb.tar_bundle({"events/1.json": b"[]"}),
        statsbomb.season_slug("lineups"): statsbomb.tar_bundle({"lineups/1.json": b"[]"}),
    }
    rows = {
        slug: {"id": i, "sha256": fb.sha256_bytes(data), "storage_path": f"datasets/{slug}.gz"}
        for i, (slug, data) in enumerate(bundles.items())
    }
    eq = client.table.return_value.select.return_value.eq
    eq.side_effect = lambda col, slug: MagicMock(**{"execute.return_value.data": [rows[slug]]})
    client.storage.from_.return_value.download.side_effect = lambda path: gzip.compress(
        bundles[path.removesuffix(".gz")]
    )
    monkeypatch.setattr(fb, "sync_experiments", lambda c: {fb.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(fb, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(fb, "start_run", lambda c, **kw: 11)
    monkeypatch.setattr(fb, "tracked", lambda c, run_id: MagicMock())
    seen = {}
    monkeypatch.setattr(fb, "compute", lambda m, e, lu, cfg: seen.update(m=m, e=e, lu=lu) or [])
    monkeypatch.setattr(fb, "write_results", lambda c, run_id, r: None)
    assert fb.run(client) == 11
    assert seen == {"m": MATCHES, "e": {1: []}, "lu": {1: []}}
    rows[fb.MATCHES_SLUG]["sha256"] = "0" * 64
    try:
        fb.run(client)
    except RuntimeError as e:
        assert "sha256" in str(e)
    else:
        raise AssertionError("tampered file accepted")


def test_possession_transitions_follow_the_holder_and_end_states():
    end_index, _ = fb.window_end(EVENTS, "Arsenal")
    moves = fb.possession_transitions(EVENTS, "Arsenal", {"A", "B", "C", "D"}, end_index)
    names = [
        (a if isinstance(a, str) else a[0], b if isinstance(b, str) else b[0]) for a, b in moves
    ]
    # Possession 1: passes and loose-ball changes of holder, then Chelsea have the ball (Lost).
    # Possession 3 is still open when the substitution ends the window, so it is left out.
    assert names == [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"),
                     ("A", "Lost")]  # fmt: skip
    assert moves[0] == (("A", "D"), ("B", "D"))  # x = 0 and x = 10 are both in the first third


def test_possession_shot_restart_and_lost():
    shot = ev("B", kind="Shot") | {"location": [100.0, 40.0]}
    events = with_possessions(
        numbered(
            [
                starting_xi(),
                ev("A", "B"),
                shot,
                ev("A", "B"),
                ev("B", "A"),
                ev("C", "D", team="Chelsea"),
            ]
        ),  # fmt: skip
        [(3, "Arsenal"), (4, "Arsenal"), (5, "Arsenal"), (9, "Chelsea")],
    )
    moves = fb.possession_transitions(events, "Arsenal", {"A", "B"}, float("inf"))
    a, b = ("A", "D"), ("B", "D")
    assert moves == [(a, b), (b, "Shot"), (a, b), (b, "Restart"), (b, a), (a, "Lost")]
    assert fb.third_of(39.9) == "D" and fb.third_of(40) == "M" and fb.third_of(80) == "A"


def test_scale_report_flags_the_one_group_baseline():
    counts = np.array([[0, 3, 1, 1], [2, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    report = fb.scale_report(counts, 2, {"pairs": [[0], [1]]}, 10, random.Random(0))
    assert report["pairs"]["ce"] == pytest.approx(0.0)
    assert set(report) == {"pairs", "one_group", "micro"}
    assert np.isnan(report["one_group"]["vs_random_pct"])
