"""AppTest smoke tests: navigation shape and every registered page renders."""

import json
from pathlib import Path

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


def test_home_shows_empty_state():
    at = AppTest.from_file(APP).run()
    assert at.title[0].value == "Ayden's Lab"
    assert any("No entries yet" in i.value for i in at.info)
