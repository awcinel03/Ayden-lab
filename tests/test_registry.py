"""The registry and app_pages/ must agree, and identifiers must be unique."""

from pathlib import Path

from lab.registry import ENTRIES, HOME_PATH, Category

ROOT = Path(__file__).resolve().parent.parent


def registered_paths() -> set[str]:
    return {HOME_PATH} | {c.index_path for c in Category} | {e.path for e in ENTRIES}


def test_every_registered_path_exists():
    missing = sorted(p for p in registered_paths() if not (ROOT / p).is_file())
    assert not missing, f"registry points to missing files: {missing}"


def test_no_orphan_page_files():
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "app_pages").rglob("*.py")}
    orphans = sorted(on_disk - registered_paths())
    assert not orphans, f"page files not in the registry: {orphans}"


def test_slugs_and_url_paths_are_unique():
    url_paths = [c.url_path for c in Category] + [e.slug for e in ENTRIES]
    assert len(url_paths) == len(set(url_paths))


def test_entries_live_in_their_category_directory():
    for e in ENTRIES:
        assert e.path.startswith(f"app_pages/{e.category.key}/"), e.slug
