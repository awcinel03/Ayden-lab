"""Shared page-building helpers. Pages call these instead of hand-rolling layout."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from lab.registry import Category, Entry, Status, published_entries

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


def category_index(category: Category) -> None:
    """Render a category's index page: title, description and its entries."""
    st.title(category.value)
    st.write(category.description)
    entries = published_entries(category)
    if entries:
        entry_list(entries, show_category=False)
    else:
        empty_state(f"No {category.value} entries yet.")
