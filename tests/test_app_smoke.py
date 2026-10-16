"""AppTest smoke tests: navigation shape and every registered page renders."""

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from lab.registry import HOME_PATH, Category, published_entries

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
PAGES = [HOME_PATH] + [c.index_path for c in Category] + [e.path for e in published_entries()]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP).run()
    if page != HOME_PATH:
        at.switch_page(page).run()
    assert not at.exception, at.exception


def test_navigation_sections_and_url_paths():
    def probe():
        import streamlit as st

        from lab.nav import build_navigation

        st.json({k: [p.url_path for p in v] for k, v in build_navigation().items()})

    at = AppTest.from_function(probe).run()
    assert not at.exception, at.exception
    nav = json.loads(at.json[0].value)
    assert list(nav) == [""] + [c.value for c in Category]
    assert nav[""] == [""]  # Home is the default page at the root URL
    for c in Category:
        assert nav[c.value][0] == c.url_path


def test_page_header_renders():
    def header():
        import datetime

        from lab.registry import Category, Entry, EntryType
        from lab.ui import page_header

        page_header(
            Entry(
                slug="demo",
                title="Demo entry",
                category=Category.NETWORK_SCIENCE,
                type=EntryType.EXPERIMENT,
                path="app_pages/network_science/demo.py",
                summary="A demo.",
                date=datetime.date(2026, 10, 15),
                tags=("graphs",),
            )
        )

    at = AppTest.from_function(header).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Demo entry"


def test_home_lists_published_entries():
    at = AppTest.from_file(APP).run()
    assert at.title[0].value == "Ayden's Lab"
    assert not at.info  # entries exist, so no empty state
    assert len(published_entries()) >= 1


def test_entry_page_without_data_shows_empty_state():
    at = AppTest.from_file(APP).run()
    at.switch_page("app_pages/network_science/graph_stats.py").run()
    assert not at.exception, at.exception
    assert any("No published results yet" in i.value for i in at.info)


def test_entry_page_renders_results(monkeypatch):
    from lab import db

    run = db.Run(
        id=1,
        experiment_slug="email-eu-core-stats",
        started_at=datetime.datetime(2026, 10, 15, tzinfo=datetime.UTC),
        finished_at=datetime.datetime(2026, 10, 15, 0, 1, tzinfo=datetime.UTC),
        config={"version": 1},
        git_sha="0" * 40,
    )
    scalars = {
        "nodes": 4,
        "edges": 5,
        "density": 0.4,
        "reciprocity": 0.6,
        "mean_in_degree": 1.25,
        "largest_wcc_fraction": 1.0,
        "largest_scc_fraction": 0.75,
        "average_clustering": 0.5,
        "transitivity": 0.4,
        "max_total_degree": 4,
    }
    rows = [{"metric": k, "value": v, "unit": None, "dims": {}} for k, v in scalars.items()]
    rows += [
        {"metric": "degree_count", "value": c, "unit": None, "dims": {"kind": k, "degree": d}}
        for k in ("total", "in", "out")
        for d, c in ((0, 1), (2, 2), (4, 1))
    ]
    dataset = db.Dataset(
        slug="snap-email-eu-core",
        name="email-Eu-core network",
        source="Stanford SNAP",
        source_url="https://snap.stanford.edu/data/email-Eu-core.html",
        license=None,
        citation="Leskovec et al.",
        sha256="ab" * 32,
        retrieved_at=datetime.datetime(2026, 10, 15, tzinfo=datetime.UTC),
        row_count=5,
    )
    monkeypatch.setattr(db, "latest_run", lambda slug: run)
    monkeypatch.setattr(db, "results", lambda run_id: pd.DataFrame(rows))
    monkeypatch.setattr(db, "run_datasets", lambda run_id: [dataset])

    at = AppTest.from_file(APP).run()
    at.switch_page("app_pages/network_science/graph_stats.py").run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric][:2] == ["4", "5"]
    assert any(h.value == "Provenance" for h in at.header)
    at.segmented_control(key="degree_kind").set_value("in").run()
    assert not at.exception, at.exception


def test_enron_page_renders_results(monkeypatch):
    from lab import db, storage

    run = db.Run(
        id=2,
        experiment_slug="enron-email-network",
        started_at=datetime.datetime(2026, 10, 15, tzinfo=datetime.UTC),
        finished_at=None,
        config={"version": 1},
        git_sha="0" * 40,
    )
    scalars = {
        "nodes": 14, "edges": 32, "giant_nodes": 12, "max_degree": 6, "avg_degree": 4.6,
        "power_law_slope": -1.5, "power_law_intercept": 1.0, "power_law_r2": 0.9,
        "power_law_points": 2, "avg_path_sampled": 2.0, "random_path": 1.9,
        "avg_clustering": 0.8, "random_clustering": 0.4, "clustering_ratio": 2.0,
        "hub_removal_loss": 6, "random_removal_loss": 1, "damage_ratio": 6.0,
        "communities": 2, "modularity": 0.45, "within_fraction": 0.97, "median_community_size": 6,
    }  # fmt: skip
    rows = [{"metric": k, "value": v, "unit": None, "dims": {}} for k, v in scalars.items()]
    rows += [{"metric": "degree_count", "value": c, "unit": None, "dims": {"degree": d}}
             for d, c in ((1, 2), (5, 10), (6, 2))]  # fmt: skip
    curves = (
        {"strategy": "targeted"},
        {"strategy": "random", "seed": 42},
        {"strategy": "random", "seed": 43},
    )
    rows += [
        {"metric": "giant_fraction", "value": 1 - 0.1 * i, "unit": None, "dims": d | {"step": i}}
        for d in curves
        for i in range(3)
    ]
    rows += [
        {"metric": "community_size", "value": 6, "unit": None, "dims": {"rank": r}} for r in (1, 2)
    ]
    rows += [{"metric": f"sweep_{m}", "value": 0.5, "unit": None, "dims": {"seed": 0}}
             for m in ("communities", "modularity", "within_fraction")]  # fmt: skip
    rows += [{"metric": "cross_community_edges", "value": 1, "unit": None,
              "dims": {"rank": 1, "a": 0, "b": 1, "a_size": 6, "b_size": 6}}]  # fmt: skip
    tables = {
        "community_nodes.parquet": pd.DataFrame(
            {"community": [0, 1], "size": [6, 6], "x": [0.0, 1.0], "y": [0.0, 1.0]}
        ),
        "community_edges.parquet": pd.DataFrame({"source": [0], "target": [1], "weight": [1]}),
    }
    arts = [db.Artifact("table", f"enron-email-network/2/{n}", 10, None) for n in tables]
    monkeypatch.setattr(db, "latest_run", lambda slug: run)
    monkeypatch.setattr(db, "results", lambda run_id: pd.DataFrame(rows))
    monkeypatch.setattr(db, "run_datasets", lambda run_id: [])
    monkeypatch.setattr(db, "artifacts", lambda run_id: arts)
    monkeypatch.setattr(
        storage, "read_parquet", lambda a: tables[a.storage_path.rsplit("/", 1)[-1]]
    )

    at = AppTest.from_file(APP).run()
    at.switch_page("app_pages/network_science/enron.py").run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric][:2] == ["14", "32"]
    assert any(h.value == "Takeaways" for h in at.header)
    assert not at.info  # community network rendered, no empty state


def test_c_elegans_page_renders_results(monkeypatch):
    from lab import db, storage

    run = db.Run(
        id=3,
        experiment_slug="c-elegans-connectome",
        started_at=datetime.datetime(2026, 10, 15, tzinfo=datetime.UTC),
        finished_at=None,
        config={"version": 1},
        git_sha="0" * 40,
    )
    scalars = {
        "nodes": 3, "edges": 4, "mean_degree": 2.67, "median_degree": 3, "max_degree": 3,
        "power_law_slope": 0.1, "power_law_intercept": 0.2, "power_law_r2": 0.01,
        "avg_path_sampled": 1.2, "random_path": 1.1, "avg_clustering": 0.5,
        "random_clustering": 0.6, "clustering_ratio": 0.8, "signal_toward_centre": 1,
        "signal_away_from_centre": 3, "communities": 1, "modularity": 0.0, "within_fraction": 1.0,
    }  # fmt: skip
    rows = [{"metric": k, "value": v, "unit": None, "dims": {}} for k, v in scalars.items()]
    rows += [
        {"metric": "degree_count", "value": c, "unit": None, "dims": {"degree": d}}
        for d, c in ((2, 2), (3, 2))
    ]
    rows += [
        {"metric": "role_count", "value": 1, "unit": None, "dims": {"role": r}}
        for r in ("integrator", "broadcaster", "relay")
    ]
    curves = [
        ({"strategy": "targeted"}, 31),
        ({"strategy": "in_degree"}, 3),
        ({"strategy": "out_degree"}, 3),
        ({"strategy": "random", "seed": 42}, 31),
        ({"strategy": "random", "seed": 43}, 31),
    ]
    rows += [
        {"metric": "giant_fraction", "value": 1 - 0.01 * i, "unit": None, "dims": d | {"step": i}}
        for d, n in curves
        for i in range(n)
    ]
    rows += [{"metric": "community_size", "value": 3, "unit": None, "dims": {"rank": 1}}]
    rows += [
        {
            "metric": "hub_community",
            "value": 3,
            "unit": None,
            "dims": {"rank": 1, "neuron": "AVAL", "community": 0, "community_size": 3},
        }
    ]
    rows += [
        {"metric": f"sweep_{m}", "value": 1, "unit": None, "dims": {"seed": 0}}
        for m in ("communities", "modularity", "within_fraction")
    ]
    tables = {
        "neurons.parquet": pd.DataFrame(
            {
                "neuron": ["AVAL", "RIH", "RIAR"],
                "x": [0.0, 1.0, 0.5],
                "y": [0.0, 0.0, 1.0],
                "in_degree": [0, 2, 2],
                "out_degree": [2, 0, 2],
                "total_degree": [2, 2, 4],
                "role": ["broadcaster", "integrator", "relay"],
                "community": [0, 0, 0],
                "hub": [False, False, True],
            }
        ),
        "synapses.parquet": pd.DataFrame(
            {
                "source": ["AVAL", "AVAL", "RIAR", "RIAR"],
                "target": ["RIH", "RIAR", "RIH", "AVAL"],
                "toward_centre": [True, False, False, False],
            }
        ),
    }
    arts = [db.Artifact("table", f"c-elegans-connectome/3/{n}", 10, None) for n in tables]
    monkeypatch.setattr(db, "latest_run", lambda slug: run)
    monkeypatch.setattr(db, "results", lambda run_id: pd.DataFrame(rows))
    monkeypatch.setattr(db, "run_datasets", lambda run_id: [])
    monkeypatch.setattr(db, "artifacts", lambda run_id: arts)
    monkeypatch.setattr(
        storage, "read_parquet", lambda a: tables[a.storage_path.rsplit("/", 1)[-1]]
    )

    at = AppTest.from_file(APP).run()
    at.switch_page("app_pages/network_science/c_elegans.py").run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric][:2] == ["3", "4"]
    assert any(h.value == "Takeaways" for h in at.header)
    assert not at.info  # neuron map rendered, no empty state


def test_football_page_renders_results(monkeypatch):
    from lab import db

    run = db.Run(
        id=4,
        experiment_slug="arsenal-passing-networks",
        started_at=datetime.datetime(2026, 10, 16, tzinfo=datetime.UTC),
        finished_at=None,
        config={"version": 1},
        git_sha="0" * 40,
    )
    rows = [
        {"metric": "match_info", "value": m, "unit": None,
         "dims": {"match": m, "date": "2003-10-18", "home": "Arsenal", "away": "Chelsea",
                  "home_score": 2, "away_score": 1}}
        for m in (1, 2)
    ]  # fmt: skip
    rows += [
        {"metric": k, "value": v, "unit": None, "dims": {"match": m}}
        for m in (1, 2)
        for k, v in (("completed_passes", 7), ("players", 2), ("connections", 2),
                     ("average_degree", 2.0))
    ]  # fmt: skip
    rows += [
        {"metric": "passes_made", "value": n, "unit": None,
         "dims": {"match": m, "player": p, "label": p, "position": pos, "x": x, "y": 10.0,
                  "received": 7 - n}}
        for m in (1, 2)
        for p, pos, n, x in (("Cole", "Left Back", 4, 40.0), ("Pires", "Left Midfield", 3, 70.0))
    ]  # fmt: skip
    rows += [
        {"metric": "pass_link", "value": n, "unit": None,
         "dims": {"match": m, "passer": a, "recipient": b}}
        for m in (1, 2)
        for a, b, n in (("Cole", "Pires", 4), ("Pires", "Cole", 3))
    ]  # fmt: skip
    monkeypatch.setattr(db, "latest_run", lambda slug: run)
    monkeypatch.setattr(db, "results", lambda run_id: pd.DataFrame(rows))
    monkeypatch.setattr(db, "run_datasets", lambda run_id: [])

    at = AppTest.from_file(APP).run()
    at.switch_page("app_pages/network_science/football.py").run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "7"
    assert any(h.value == "Takeaways" for h in at.header)
    assert "flanks" in at.markdown[-1].value or any("flanks" in m.value for m in at.markdown)
    at.slider(key="min_passes").set_value(1).run()
    assert not at.exception, at.exception


def test_football_logo_assets_exist():
    root = Path(__file__).resolve().parent.parent / "lab" / "assets"
    for theme in ("light", "dark"):
        assert (root / f"statsbomb-logo-{theme}.png").is_file()


def test_coral_page_renders_results(monkeypatch):
    from lab import db, storage

    run = db.Run(
        id=5,
        experiment_slug="coral-heat-stress-forecast",
        started_at=datetime.datetime(2026, 10, 16, tzinfo=datetime.UTC),
        finished_at=None,
        config={"version": 1},
        git_sha="0" * 40,
    )
    scalars = {
        "rmse": 1.2, "mae": 0.6, "r2": 0.83, "skill_vs_persistence": 0.37, "mean_late_days": 14.7,
        "n_train": 100, "n_test": 50, "embargoed_rows": 5, "grid_cells": 2,
        "rmse_baseline_mean": 3.1, "rmse_baseline_persistence": 1.9, "best_shift_days": -8,
        "rmse_at_best_shift": 0.6, "rmse_unshifted": 0.7,
    }  # fmt: skip
    rows = [{"metric": k, "value": v, "unit": None, "dims": {}} for k, v in scalars.items()]
    rows += [{"metric": "feature_importance", "value": v, "unit": None, "dims": {"feature": f}}
             for f, v in (("dhw_lag0", 0.6), ("dhw_accel7", 0.2))]  # fmt: skip
    for fs, n, late in (("sst_ssta", 8, None), ("lags", 12, 16.0), ("full", 15, 14.7)):
        for m, v in (
            ("n_features", n),
            ("rmse", 1.2),
            ("r2", 0.8),
            ("skill_vs_persistence", 0.3),
            ("best_shift", -8),
            ("mean_late", late),
        ):
            rows.append({"metric": f"ablation_{m}", "value": v, "unit": None,
                         "dims": {"feature_set": fs}})  # fmt: skip
    rows += [{"metric": "shift_rmse", "value": 1 + abs(s) / 10, "unit": None,
              "dims": {"shift": s}} for s in range(-2, 3)]  # fmt: skip
    rows += [
        {"metric": "crossing_late_days", "value": 17.0, "unit": None,
         "dims": {"year": 2020, "actual": "2020-02-21", "predicted": "2020-03-09"}},
        {"metric": "crossing_late_days", "value": None, "unit": None,
         "dims": {"year": 2023, "actual": None, "predicted": "2023-01-12"}},
    ]  # fmt: skip
    days = pd.to_datetime(["2020-02-01", "2020-02-02"])
    tables = {
        "daily.parquet": pd.DataFrame(
            {"target_date": days, "actual": [3.0, 5.0], "predicted": [2.0, 4.5],
             "persistence": [1.0, 3.0]}
        ),
        "cells.parquet": pd.DataFrame(
            {"target_date": days.repeat(2), "latitude": [-10.0, -11.0] * 2,
             "longitude": [142.0, 143.0] * 2, "actual": [0.5, 9.0, 2.0, 5.0],
             "predicted": [-0.1, 8.5, 2.0, 5.0]}
        ),
    }  # fmt: skip
    arts = [db.Artifact("table", f"coral-heat-stress-forecast/5/{n}", 10, None) for n in tables]
    monkeypatch.setattr(db, "latest_run", lambda slug: run)
    monkeypatch.setattr(db, "results", lambda run_id: pd.DataFrame(rows))
    monkeypatch.setattr(db, "run_datasets", lambda run_id: [])
    monkeypatch.setattr(db, "artifacts", lambda run_id: arts)
    monkeypatch.setattr(
        storage, "read_parquet", lambda a: tables[a.storage_path.rsplit("/", 1)[-1]].copy()
    )

    at = AppTest.from_file(APP).run()
    at.switch_page("app_pages/machine_learning/coral_forecast.py").run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "1.200 °C-wk"
    assert not at.info  # every artifact rendered
    assert any("late every time" in m.value for m in at.markdown)
    at.select_slider(key="map_date").set_value(days[0].date()).run()
    assert not at.exception, at.exception


def _wine_fixtures():
    from pipelines.experiments.network_science import wine_flavor as wx
    from tests.test_pipelines_wine_flavor import SMALL, toy_tables

    terms, edges = toy_tables()
    v2_terms = terms.assign(substring_only_reviews=0, forms_only_reviews=2)
    rows, flavors, cooccurrence, paths = wx.compute(v2_terms, edges, SMALL)
    # The original (v1) matching: one fewer bridge and inflated counts.
    rows += wx.compute_comparison(terms.assign(reviews=terms["reviews"] + 5), edges[1:], SMALL)
    results = pd.DataFrame([r | {"dims": r.get("dims", {})} for r in rows])
    tables = {
        "flavors.parquet": pd.DataFrame(flavors),
        "cooccurrence.parquet": pd.DataFrame(cooccurrence),
        "shortest_paths.parquet": pd.DataFrame(paths),
    }
    return results, tables


@pytest.mark.parametrize("with_artifacts", [True, False])
def test_wine_flavor_page_renders_results(monkeypatch, with_artifacts):
    from lab import db, storage

    run = db.Run(
        id=4,
        experiment_slug="wine-flavor-network",
        started_at=datetime.datetime(2026, 10, 16, tzinfo=datetime.UTC),
        finished_at=None,
        config={"version": 1},
        git_sha="0" * 40,
    )
    results, tables = _wine_fixtures()
    arts = [db.Artifact("table", f"wine-flavor-network/4/{n}", 10, None) for n in tables]
    monkeypatch.setattr(db, "latest_run", lambda slug: run)
    monkeypatch.setattr(db, "results", lambda run_id: results)
    monkeypatch.setattr(db, "run_datasets", lambda run_id: [])
    monkeypatch.setattr(db, "artifacts", lambda run_id: arts if with_artifacts else [])
    monkeypatch.setattr(
        storage, "read_parquet", lambda a: tables[a.storage_path.rsplit("/", 1)[-1]]
    )

    at = AppTest.from_file(APP).run()
    at.switch_page("app_pages/network_science/wine_flavor.py").run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric][:2] == ["10", "15"]
    assert any(h.value == "Takeaways" for h in at.header)
    page_text = " ".join(m.value for m in at.markdown)
    assert "brett (17 reviews)" in page_text
    assert any(h.value == "What changed from my original analysis" for h in at.header)
    assert len(at.dataframe) >= 3  # pairings, change table, v1-vs-v2 table
    if with_artifacts:
        assert not at.info
        assert "I read this as the red/white divide" in page_text
        # Isolated pair: the picker says there is no path.
        at.selectbox[0].set_value("brett").run()
        assert not at.exception, at.exception
        assert any("No path" in m.value for m in at.markdown)
        at.selectbox[0].set_value("lemon").run()
        at.selectbox[1].set_value("plum").run()
        assert any("**lemon** →" in m.value and "(4 hops)" in m.value for m in at.markdown)
    else:
        assert at.info  # empty states for the network and the path picker
