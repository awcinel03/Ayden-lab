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
