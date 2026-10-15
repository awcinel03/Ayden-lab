"""Single source of truth for every page in the lab.

Pure Python with no Streamlit import, so pipelines and tests can import it.
Adding a page = one file under ``app_pages/`` + one ``Entry`` in ``ENTRIES``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    """Sidebar sections. Declaration order is sidebar order; value is the label."""

    NETWORK_SCIENCE = "Network Science"
    MACHINE_LEARNING = "Machine Learning"
    CLAUDE_API = "Claude API"
    MODEL_TESTING = "Model Testing"
    AI_GENERAL = "AI General"

    @property
    def key(self) -> str:
        """Snake-case key: the ``app_pages/`` directory and ``experiments.category`` value."""
        return self.name.lower()

    @property
    def url_path(self) -> str:
        """URL path of the category index page, e.g. ``network-science``."""
        return self.key.replace("_", "-")

    @property
    def index_path(self) -> str:
        """Path of the category index page, relative to the repo root."""
        return f"app_pages/{self.key}/index.py"

    @property
    def description(self) -> str:
        """One-line description shown on the index and home pages."""
        return _CATEGORY_DESCRIPTIONS[self]


_CATEGORY_DESCRIPTIONS: dict[Category, str] = {
    Category.NETWORK_SCIENCE: "Graph analysis, network models, complexity.",
    Category.MACHINE_LEARNING: "Models, training experiments, classifiers.",
    Category.CLAUDE_API: "Tools and experiments built on the Claude API.",
    Category.MODEL_TESTING: "Cross-provider evals and comparisons.",
    Category.AI_GENERAL: "Broader explorations, essays-with-data, one-offs.",
}


class EntryType(StrEnum):
    """Kind of entry, shown as a badge in the page header."""

    TOOL = "Tool"
    EXPERIMENT = "Experiment"
    FINDING = "Finding"
    GRAPH = "Graph"
    MODEL = "Model"
    EXPLORATION = "Exploration"


class Status(StrEnum):
    """Lifecycle status of an entry."""

    DRAFT = "Draft"
    ACTIVE = "Active"
    COMPLETE = "Complete"
    ARCHIVED = "Archived"


@dataclass(frozen=True)
class Entry:
    """Metadata for one entry page."""

    slug: str  # globally unique; used as url_path and as experiments.slug
    title: str
    category: Category
    type: EntryType
    path: str  # relative to repo root, e.g. "app_pages/network_science/graph_stats.py"
    summary: str
    date: datetime.date
    status: Status = Status.ACTIVE
    tags: tuple[str, ...] = ()
    icon: str | None = None  # Material icon, e.g. ":material/hub:"
    published: bool = True  # False hides it from nav, indexes and home


HOME_PATH = "app_pages/home.py"

ENTRIES: tuple[Entry, ...] = ()


def published_entries(category: Category | None = None) -> list[Entry]:
    """Return published entries, newest first, optionally limited to one category."""
    entries = [e for e in ENTRIES if e.published and (category is None or e.category == category)]
    return sorted(entries, key=lambda e: e.date, reverse=True)
