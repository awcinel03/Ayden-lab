"""Shared fixtures: tests never reach the live Supabase project."""

import pytest
import streamlit as st


@pytest.fixture(autouse=True)
def no_live_supabase(monkeypatch):
    """Hide any local Streamlit secrets file so lab.db sees no credentials."""
    from lab import db

    monkeypatch.setattr(db.st, "secrets", {})
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()
