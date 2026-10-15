"""Shared page-building helpers. Pages call these instead of hand-rolling layout."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from lab.db import Dataset, Run
from lab.registry import Category, Entry, Status, published_entries

REPO_URL = "https://github.com/awcinel03/Ayden-lab"

_STATUS_COLORS = {
    Status.DRAFT: "gray",
    Status.ACTIVE: "blue",
    Status.COMPLETE: "green",
    Status.ARCHIVED: "orange",
}


def page_header(entry: Entry) -> None:
    """Render an entry's title, category, type and status badges, date and tags."""
    st.title(entry.title)
    with st.container(horizontal=True):
        st.badge(entry.category.value, icon=":material/folder:", color="primary")
        st.badge(entry.type.value, color="violet")
        st.badge(entry.status.value, color=_STATUS_COLORS[entry.status])
        st.badge(entry.date.isoformat(), icon=":material/calendar_today:", color="gray")
    if entry.tags:
        st.caption(" · ".join(f"#{tag}" for tag in entry.tags))
    st.write(entry.summary)


def empty_state(message: str) -> None:
    """Show a friendly placeholder where content will appear."""
    st.info(message, icon=":material/hourglass_empty:")


def entry_list(entries: Sequence[Entry], *, show_category: bool = True) -> None:
    """Render entries as bordered cards linking to their pages."""
    for entry in entries:
        with st.container(border=True):
            st.page_link(entry.path, label=entry.title, icon=entry.icon)
            meta = [entry.type.value, entry.date.isoformat()]
            if show_category:
                meta.insert(0, entry.category.value)
            st.caption(" · ".join(meta))
            st.write(entry.summary)


def provenance_footer(run: Run, datasets: Sequence[Dataset], code_paths: Sequence[str]) -> None:
    """Render datasets, run metadata and links to the code at the run's commit."""
    st.header("Provenance")
    for d in datasets:
        with st.container(border=True):
            st.markdown(f"**{d.name}**: [{d.source}]({d.source_url})")
            st.caption(
                f"Licence: {d.license or 'not stated by the source'} · "
                f"sha256 `{d.sha256[:12]}…` · retrieved {d.retrieved_at:%Y-%m-%d}"
            )
            if d.citation:
                st.caption(f"Cite: {d.citation}")
    finished = f"{run.finished_at:%Y-%m-%d %H:%M} UTC" if run.finished_at else "unknown"
    st.markdown(f"Run **#{run.id}**, finished {finished}, commit `{run.git_sha[:7]}`")
    st.markdown(" · ".join(f"[{p}]({REPO_URL}/blob/{run.git_sha}/{p})" for p in code_paths))
    with st.expander("Run config"):
        st.json(run.config)


def category_index(category: Category) -> None:
    """Render a category's index page: title, description and its entries."""
    st.title(category.value)
    st.write(category.description)
    entries = published_entries(category)
    if entries:
        entry_list(entries, show_category=False)
    else:
        empty_state(f"No {category.value} entries yet.")
