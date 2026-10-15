"""Live RLS and grants check against the real project. Not collected by pytest.

Run: .venv/bin/python -m tests.rls_check

Creates two fixture experiments (one published, one hidden) with the secret key,
checks what the publishable key can and cannot do, then deletes the fixtures.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from pipelines.common.client import get_client
from supabase import Client, create_client

PUB, HIDDEN = "rls-check-published", "rls-check-hidden"
failures: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


def raises(fn) -> bool:
    """True if the call raised."""
    try:
        fn()
    except Exception:
        return True
    return False


def denied(fn) -> bool:
    """True if the call raised or affected no rows."""
    try:
        res = fn()
    except Exception:
        return True
    return not getattr(res, "data", None)


def setup(admin: Client) -> dict[str, int]:
    rows = (
        admin.table("experiments")
        .upsert(
            [
                {"slug": PUB, "title": "RLS check", "category": "ai_general", "published": True},
                {
                    "slug": HIDDEN,
                    "title": "RLS check",
                    "category": "ai_general",
                    "published": False,
                },
            ],
            on_conflict="slug",
        )
        .execute()
        .data
    )
    ids = {r["slug"]: r["id"] for r in rows}
    runs = {}
    for slug, exp_id in ids.items():
        run = (
            admin.table("runs")
            .upsert(
                {
                    "experiment_id": exp_id,
                    "status": "succeeded",
                    "config": {},
                    "git_sha": "rls-check",
                    "fingerprint": f"rls-check-{slug}",
                },
                on_conflict="fingerprint",
            )
            .execute()
            .data[0]
        )
        runs[slug] = run["id"]
        admin.table("results").upsert(
            {"run_id": run["id"], "metric": "check", "value": 1, "dims": {}},
            on_conflict="run_id,metric,dims",
        ).execute()
        admin.storage.from_("artifacts").upload(
            f"{slug}/0/check.txt",
            b"ok",
            file_options={"content-type": "text/plain", "upsert": "true"},
        )
    return runs


def teardown(admin: Client) -> None:
    admin.storage.from_("artifacts").remove([f"{s}/0/check.txt" for s in (PUB, HIDDEN)])
    exp_ids = [
        r["id"]
        for r in admin.table("experiments").select("id").in_("slug", [PUB, HIDDEN]).execute().data
    ]
    if exp_ids:
        admin.table("runs").delete().in_("experiment_id", exp_ids).execute()
        admin.table("experiments").delete().in_("id", exp_ids).execute()


def main() -> int:
    load_dotenv()
    admin = get_client()
    anon = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_PUBLISHABLE_KEY"])
    runs = setup(admin)
    try:
        slugs = {r["slug"] for r in anon.table("experiments").select("slug").execute().data}
        check("anon reads published experiment", PUB in slugs)
        check("anon cannot see unpublished experiment", HIDDEN not in slugs)

        visible = {r["id"] for r in anon.table("runs").select("id").execute().data}
        check("anon reads run of published experiment", runs[PUB] in visible)
        check("anon cannot see run of unpublished experiment", runs[HIDDEN] not in visible)

        res = anon.table("results").select("run_id").in_("run_id", list(runs.values())).execute()
        check("results follow run visibility", {r["run_id"] for r in res.data} == {runs[PUB]})

        latest = anon.table("latest_runs").select("experiment_slug").execute().data
        check(
            "latest_runs respects RLS",
            {r["experiment_slug"] for r in latest} >= {PUB}
            and HIDDEN not in {r["experiment_slug"] for r in latest},
        )

        check(
            "anon cannot insert",
            denied(
                lambda: (
                    anon.table("experiments")
                    .insert({"slug": "rls-check-anon", "title": "x", "category": "ai_general"})
                    .execute()
                )
            ),
        )
        check(
            "anon cannot update",
            denied(
                lambda: (
                    anon.table("experiments").update({"title": "hacked"}).eq("slug", PUB).execute()
                )
            ),
        )
        check(
            "anon cannot delete",
            denied(lambda: anon.table("results").delete().eq("run_id", runs[PUB]).execute()),
        )
        still = admin.table("experiments").select("title").eq("slug", PUB).execute().data
        check(
            "published row unchanged after anon writes", still and still[0]["title"] == "RLS check"
        )

        bucket = anon.storage.from_("artifacts")
        check(
            "anon downloads artifact of published experiment",
            not raises(lambda: bucket.download(f"{PUB}/0/check.txt")),
        )
        check(
            "anon cannot download artifact of unpublished experiment",
            raises(lambda: bucket.download(f"{HIDDEN}/0/check.txt")),
        )
        check(
            "anon cannot upload",
            raises(
                lambda: bucket.upload(
                    f"{PUB}/0/anon.txt", b"x", file_options={"content-type": "text/plain"}
                )
            ),
        )
        check("anon cannot list datasets bucket", not anon.storage.from_("datasets").list())
    finally:
        teardown(admin)
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
