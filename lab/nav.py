"""Builds ``st.navigation`` input from the registry. The only place ``st.Page`` is created."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from lab.registry import HOME_PATH, Category, published_entries

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_navigation() -> dict[str, list[st.Page]]:
    """Return ``{section label: [pages]}`` with Home first and every category present.

    Every page gets an explicit ``url_path``: all index files are named ``index.py``
    and Streamlit does not support nested URL paths, so inferred paths would collide.
    """
    nav: dict[str, list[st.Page]] = {
        "": [st.Page(REPO_ROOT / HOME_PATH, title="Home", icon=":material/home:", default=True)]
    }
    for category in Category:
        pages = [
            st.Page(
                REPO_ROOT / category.index_path,
                title="Overview",
                icon=":material/list:",
                url_path=category.url_path,
            )
        ]
        pages += [
            st.Page(REPO_ROOT / entry.path, title=entry.title, icon=entry.icon, url_path=entry.slug)
            for entry in published_entries(category)
        ]
        nav[category.value] = pages
    return nav
