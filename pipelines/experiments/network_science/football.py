"""Arsenal passing networks: who passes to whom, and from where, in two 2003/04 matches.

Run: python -m pipelines.experiments.network_science.football [--allow-dirty]

A port of my football_networks notebook. Loads the StatsBomb matches, events and
lineups files from Storage (verifying every sha256), builds a directed passing
network per match from Arsenal's completed passes, places each player at their
average on-ball position, and writes long-format results. No raw event data is
stored in results, only the derived network.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np

from pipelines.common.client import get_client
from pipelines.common.provenance import git_sha, sha256_bytes
from pipelines.common.runs import start_run, tracked, write_results
from pipelines.common.sync import sync_experiments
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "arsenal-passing-networks"
MATCHES_SLUG = "statsbomb-pl-2003-04-matches"
MATCH_IDS = (3749079, 3749552)  # Arsenal 2-1 Chelsea; Man United 0-0 Arsenal
CONFIG: dict[str, Any] = {
    "version": 1,
    "datasets": [MATCHES_SLUG]
    + [f"statsbomb-{kind}-{m}" for m in MATCH_IDS for kind in ("events", "lineups")],
    "team": "Arsenal",
    "matches": list(MATCH_IDS),
    "completed_pass": "type Pass by the team with no pass.outcome",
    "edge": "passer -> recipient, weight = completed passes; passes with no recipient skipped",
    "average_position": (
        "mean of the player's completed-pass start locations and the end locations of "
        "completed passes they received"
    ),
    "position": "the position a player made most completed passes from",
    "label": "lineup nickname, else full name",
    "average_degree": "sum of in + out degree / players (unweighted)",
    "draw_min_passes": 3,
}


def completed_passes(events: list[dict[str, Any]], team: str) -> list[dict[str, Any]]:
    """The team's passes with no ``pass.outcome`` (StatsBomb leaves it out when completed)."""
    return [
        e
        for e in events
        if e["type"]["name"] == "Pass" and e["team"]["name"] == team and "outcome" not in e["pass"]
    ]


def passing_network(passes: list[dict[str, Any]]) -> nx.DiGraph:
    """Directed graph, passer → recipient, weighted by completed passes."""
    g = nx.DiGraph()
    for e in passes:
        recipient = e["pass"].get("recipient")
        if recipient is None:
            continue
        u, v = e["player"]["name"], recipient["name"]
        if g.has_edge(u, v):
            g[u][v]["weight"] += 1
        else:
            g.add_edge(u, v, weight=1)
    return g


def average_positions(passes: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Each player's mean on-ball location, as in the notebook's ``plot_passing_network``."""
    xs: dict[str, list[float]] = defaultdict(list)
    ys: dict[str, list[float]] = defaultdict(list)
    for e in passes:
        player = e["player"]["name"]
        xs[player].append(e["location"][0])
        ys[player].append(e["location"][1])
        recipient, end = e["pass"].get("recipient"), e["pass"].get("end_location")
        if recipient is not None and end is not None:
            xs[recipient["name"]].append(end[0])
            ys[recipient["name"]].append(end[1])
    return {p: (float(np.mean(xs[p])), float(np.mean(ys[p]))) for p in xs}


def main_positions(passes: list[dict[str, Any]]) -> dict[str, str]:
    """Position each player made most completed passes from (first seen wins ties)."""
    counts = Counter((e["player"]["name"], e["position"]["name"]) for e in passes)
    best: dict[str, tuple[int, str]] = {}
    for (player, position), n in counts.items():
        if player not in best or n > best[player][0]:
            best[player] = (n, position)
    return {p: pos for p, (_, pos) in best.items()}


def labels(lineups: list[dict[str, Any]], team: str) -> dict[str, str]:
    """Display name per player: the lineup nickname when there is one."""
    return {
        p["player_name"]: p["player_nickname"] or p["player_name"]
        for t in lineups
        if t["team_name"] == team
        for p in t["lineup"]
    }


def compute(
    matches: list[dict[str, Any]],
    events: dict[int, list[dict[str, Any]]],
    lineups: dict[int, list[dict[str, Any]]],
    config: dict[str, Any] = CONFIG,
) -> list[dict[str, Any]]:
    """Long-format result rows for every configured match."""
    team = config["team"]
    by_id = {m["match_id"]: m for m in matches}
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float, unit: str | None = None, **dims: Any) -> None:
        rows.append({"metric": metric, "value": value, "unit": unit, "dims": dims})

    for match_id in config["matches"]:
        m = by_id[match_id]
        add(
            "match_info",
            match_id,
            match=match_id,
            date=m["match_date"],
            home=m["home_team"]["home_team_name"],
            away=m["away_team"]["away_team_name"],
            home_score=m["home_score"],
            away_score=m["away_score"],
        )
        passes = completed_passes(events[match_id], team)
        g = passing_network(passes)
        n = g.number_of_nodes()
        add("completed_passes", len(passes), "passes", match=match_id)
        add("players", n, "players", match=match_id)
        add("connections", g.number_of_edges(), "edges", match=match_id)
        add("average_degree", sum(d for _, d in g.degree()) / n if n else 0.0, match=match_id)

        made = Counter(e["player"]["name"] for e in passes)
        received = Counter(
            e["pass"]["recipient"]["name"] for e in passes if "recipient" in e["pass"]
        )
        pos, where, names = (
            average_positions(passes),
            main_positions(passes),
            labels(lineups[match_id], team),
        )
        for player in g.nodes():
            x, y = pos[player]
            add(
                "passes_made",
                made[player],
                "passes",
                match=match_id,
                player=player,
                label=names.get(player, player),
                position=where.get(player),
                x=x,
                y=y,
                received=received[player],
            )
        for u, v, d in g.edges(data=True):
            add("pass_link", d["weight"], "passes", match=match_id, passer=u, recipient=v)
    return rows


def _download(client: Client, dataset: dict[str, Any]) -> Any:
    bucket, path = dataset["storage_path"].split("/", 1)
    data = client.storage.from_(bucket).download(path)
    if sha256_bytes(data) != dataset["sha256"]:
        raise RuntimeError(f"{path} does not match its recorded sha256")
    return json.loads(data)


def run(client: Client, *, allow_dirty: bool = False) -> int | None:
    """Execute the experiment; returns the run id, or ``None`` if skipped."""
    experiment_id = sync_experiments(client)[EXPERIMENT_SLUG]
    datasets = {}
    for slug in CONFIG["datasets"]:
        rows = (
            client.table("datasets")
            .select("id, sha256, storage_path")
            .eq("slug", slug)
            .execute()
            .data
        )
        if not rows:
            raise RuntimeError(f"Dataset {slug} missing; run pipelines.ingest.statsbomb first")
        datasets[slug] = rows[0]
    run_id = start_run(
        client,
        experiment_id=experiment_id,
        config=CONFIG,
        git_sha=git_sha(allow_dirty=allow_dirty),
        datasets={d["id"]: d["sha256"] for d in datasets.values()},
    )
    if run_id is None:
        return None
    with tracked(client, run_id):
        matches = _download(client, datasets[MATCHES_SLUG])
        events = {m: _download(client, datasets[f"statsbomb-events-{m}"]) for m in MATCH_IDS}
        lineups = {m: _download(client, datasets[f"statsbomb-lineups-{m}"]) for m in MATCH_IDS}
        write_results(client, run_id, compute(matches, events, lineups, CONFIG))
    return run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-dirty", action="store_true", help="record a run from uncommitted code"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_id = run(get_client(), allow_dirty=args.allow_dirty)
    log.info(
        "skipped: identical run already succeeded" if run_id is None else f"run {run_id} succeeded"
    )


if __name__ == "__main__":
    main()
