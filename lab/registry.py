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

ENTRIES: tuple[Entry, ...] = (
    Entry(
        slug="email-eu-core-stats",
        title="email-Eu-core: basic stats and degree distribution",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.EXPERIMENT,
        path="app_pages/network_science/graph_stats.py",
        summary=(
            "Size, connectivity, clustering and degree distribution of a small directed "
            "email network from a European research institution (SNAP)."
        ),
        date=datetime.date(2026, 10, 15),
        tags=("snap", "degree-distribution", "directed-graph"),
        icon=":material/hub:",
    ),
    Entry(
        slug="enron-email-network",
        title="Enron email network",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.EXPERIMENT,
        path="app_pages/network_science/enron.py",
        summary=(
            "How Enron's email network organised itself and why that made it fragile: "
            "scale-free hubs, small-world paths, collapse under targeted attack, and "
            "Louvain communities that recover the org chart."
        ),
        date=datetime.date(2026, 10, 15),
        tags=("snap", "scale-free", "robustness", "louvain"),
        icon=":material/mail:",
    ),
    Entry(
        slug="c-elegans-connectome",
        title="C. elegans frontal connectome",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.EXPERIMENT,
        path="app_pages/network_science/c_elegans.py",
        summary=(
            "The worm's directed neural wiring: no power law and no critical hubs, neuron "
            "roles read from signal direction, a small world that tolerates damage, and "
            "evenly sized, well-integrated circuits."
        ),
        date=datetime.date(2026, 10, 15),
        tags=("snap", "directed-graph", "connectome", "robustness", "louvain"),
        icon=":material/neurology:",
    ),
    Entry(
        slug="wine-flavor-network",
        title="Wine flavor network",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.EXPERIMENT,
        path="app_pages/network_science/wine_flavor.py",
        summary=(
            "Which descriptors 130,000 professional tasting notes use together: a near-complete "
            "co-occurrence network of 116 aroma-wheel terms with no critical hubs, robustness "
            "that is simple subtraction, and communities that split red from white."
        ),
        date=datetime.date(2026, 10, 16),
        tags=("kaggle", "text-network", "weighted-graph", "robustness", "louvain"),
        icon=":material/wine_bar:",
    ),
    Entry(
        slug="arsenal-passing-networks",
        title="Arsenal passing networks, 2003/04",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.GRAPH,
        path="app_pages/network_science/football.py",
        summary=(
            "The Invincibles' 38 league matches as passing networks: who the ball flowed through, "
            "which structure beats chance, and a Markov-chain test of whether the team is better "
            "described by players or by positions."
        ),
        date=datetime.date(2026, 10, 16),
        tags=(
            "statsbomb",
            "football",
            "directed-graph",
            "null-models",
            "markov-chains",
            "causal-emergence",
        ),
        icon=":material/sports_soccer:",
    ),
    Entry(
        slug="causal-emergence-worm-vs-email",
        title="Causal emergence in three networks",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.EXPERIMENT,
        path="app_pages/network_science/causal_emergence.py",
        summary=(
            "Do macro-nodes describe a network better than its individual nodes? Effective "
            "information for the C. elegans connectome, the Enron email network and Arsenal's "
            "passing, tested against random groupings and null networks."
        ),
        date=datetime.date(2026, 10, 19),
        tags=(
            "snap",
            "statsbomb",
            "connectome",
            "markov-chains",
            "causal-emergence",
            "effective-information",
            "null-models",
        ),
        icon=":material/hub:",
    ),
    Entry(
        slug="gravity-mass-information-flow",
        title="Gravity and Mass: How Heavy Information Moves",
        category=Category.NETWORK_SCIENCE,
        type=EntryType.MODEL,
        path="app_pages/network_science/gravity_mass.py",
        summary=(
            "A random walk where nodes are gravity wells and messages have mass. Heavy "
            "messages fall toward deep wells but get caught in shallow ones, so there is a "
            "best mass, set by how rough the landscape is. Explore it live, then see the "
            "full runs on synthetic graphs and a mouse connectome."
        ),
        date=datetime.date(2026, 10, 23),
        tags=(
            "random-walks",
            "hitting-time",
            "connectome",
            "contagion",
            "effective-information",
            "interactive",
        ),
        icon=":material/public:",
    ),
    Entry(
        slug="coral-heat-stress-forecast",
        title="Forecasting Great Barrier Reef heat stress",
        category=Category.MACHINE_LEARNING,
        type=EntryType.MODEL,
        path="app_pages/machine_learning/coral_forecast.py",
        summary=(
            "A Random Forest that predicts Degree Heating Weeks 28 days ahead from NOAA Coral "
            "Reef Watch satellite data: well ahead of the baselines, but it warns of bleaching "
            "about two weeks late, and the rate-of-change features barely fix that."
        ),
        date=datetime.date(2026, 10, 16),
        tags=("noaa", "random-forest", "forecasting", "time-series", "coral-reefs"),
        icon=":material/waves:",
    ),
)


def entry(slug: str) -> Entry:
    """Return the entry with ``slug``; raises ``KeyError`` if there is none."""
    for e in ENTRIES:
        if e.slug == slug:
            return e
    raise KeyError(slug)


def published_entries(category: Category | None = None) -> list[Entry]:
    """Return published entries, newest first, optionally limited to one category."""
    entries = [e for e in ENTRIES if e.published and (category is None or e.category == category)]
    return sorted(entries, key=lambda e: e.date, reverse=True)
