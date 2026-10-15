"""Service-role Supabase client for offline pipelines. Never imported by the app."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from supabase import Client, create_client


def get_client() -> Client:
    """Return a Supabase client using ``SUPABASE_URL`` and ``SUPABASE_SECRET_KEY`` from ``.env``.

    Raises ``RuntimeError`` if either is missing or the key is not a secret key.
    """
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SECRET_KEY in .env")
    if key.startswith("sb_publishable_"):
        raise RuntimeError(
            "SUPABASE_SECRET_KEY holds a publishable key; pipelines need the secret key"
        )
    return create_client(url, key)
