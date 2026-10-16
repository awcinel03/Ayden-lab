"""Arsenal passing networks across the 2003/04 season: structure, null models, Markov chains.

Run: python -m pipelines.experiments.network_science.football [--allow-dirty]

Started as a port of my football_networks notebook (two matches) and now covers
all 38 league matches of the unbeaten season. Loads the StatsBomb season
bundles from Storage (verifying every sha256) and, for every match:

1. the full-match passing network (players at their average on-ball position),
   exactly as the notebook built it;
2. a *window* network of the starting XI from kick-off to Arsenal's first
   substitution or red card, with structural metrics, bootstrap confidence
   intervals and a configuration-model null (pass volumes per player kept);
3. the window network read as a Markov chain of ball possession: effective
   information, and whether grouping players into positional roles or lines
   (causal emergence) or a greedy search finds a more informative scale.

The season is also pooled by starting-position slot. Only derived counts and
statistics are written; no event rows.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np

from pipelines.common import markov as mk
from pipelines.common.client import get_client
from pipelines.common.provenance import git_sha, sha256_bytes
from pipelines.common.runs import start_run, tracked, write_results
from pipelines.common.sync import sync_experiments
from pipelines.ingest.statsbomb import read_bundle, season_slug
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "arsenal-passing-networks"
MATCHES_SLUG = "statsbomb-pl-2003-04-matches"
FEATURED = (3749079, 3749552)  # Arsenal 2-1 Chelsea; Man United 0-0 Arsenal
RED_CARDS = {"Red Card", "Second Yellow"}
WINDOW_STATS = ("intensity", "density", "centralization", "mean_entropy", "reciprocity",
                "top_pair_share", "ei", "effectiveness", "ce_roles", "ce_lines")  # fmt: skip
NULL_STATS = ("density", "mean_entropy", "reciprocity", "top_pair_share", "ei", "ce_roles")
CONFIG: dict[str, Any] = {
    "version": 2,
    "datasets": [MATCHES_SLUG, season_slug("events"), season_slug("lineups")],
    "team": "Arsenal",
    "featured_matches": list(FEATURED),
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
    "window": (
        "starting XI only, completed passes between them from kick-off until Arsenal's first "
        "substitution or red card (straight or second yellow)"
    ),
    "metrics": {
        "intensity": "window passes / window minutes",
        "density": "ordered pairs with >= 1 pass / (n(n-1))",
        "centralization": "sum(s_max - s_i) / ((n-2) P), s = passes made + received, P = passes",
        "mean_entropy": "mean over passers of the entropy (bits) of their pass distribution",
        "reciprocity": "sum over pairs of 2 min(w_ij, w_ji) / P",
        "top_pair_share": "max over pairs of (w_ij + w_ji) / P",
        "ei": "effective information (bits) of the row-normalised pass matrix (Klein & Hoel 2020)",
        "effectiveness": "ei / log2(passers)",
        "ce": "EI(macro) - EI(micro); macro-nodes weighted by passes made (mu|pi)",
    },
    "roles": "starting position -> side (L/C/R; any 'Center' position is C) + line",
    "lines": "GK; DEF (*Back); ATT (*Forward, *Striker, *Wing); MID (everything else)",
    "bootstrap": {
        "reps": 500,
        "seed": 42,
        "resample": "passes with replacement",
        "interval": "basic (reverse percentile), 95%",
    },
    "null": {
        "model": "configuration: out- and in-pass totals kept, no self-passes",
        "reps": 500,
        "seed": 42,
    },  # fmt: skip
    "random_partitions": {"reps": 500, "seed": 42, "sizes": "same as the role partition"},
    "greedy": "agglomerative merges maximising macro EI; best partition seen",
    "season_tests": {"permutations": 10000, "seed": 42},
}


# --------------------------------------------------------------- full-match network (notebook)
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


# --------------------------------------------------------------- window network
def line_of(position: str) -> str:
    """GK, DEF, MID or ATT from a StatsBomb position name."""
    if position == "Goalkeeper":
        return "GK"
    if "Back" in position:
        return "DEF"
    if "Forward" in position or "Striker" in position or "Wing" in position:
        return "ATT"
    return "MID"


def role_of(position: str) -> str:
    """Side and line, e.g. ``L-DEF``; centre pairs (left/right centre back) share ``C``."""
    line = line_of(position)
    if line == "GK":
        return "GK"
    side = "C" if "Center" in position else "L" if position.startswith("Left") else "R"
    if not position.startswith(("Left", "Right")):
        side = "C"
    return f"{side}-{line}"


def window_counts(
    events: list[dict[str, Any]], team: str
) -> tuple[list[str], list[str], np.ndarray, int]:
    """Starting XI names and positions, their pass-count matrix, and the window's end minute.

    The window ends at the team's first substitution or red card (straight or
    second yellow), or at the end of the match.
    """
    xi = next(e for e in events if e["type"]["name"] == "Starting XI" and e["team"]["name"] == team)
    names = [p["player"]["name"] for p in xi["tactics"]["lineup"]]
    positions = [p["position"]["name"] for p in xi["tactics"]["lineup"]]

    def ends_window(e: dict[str, Any]) -> bool:
        if e["team"]["name"] != team:
            return False
        card = (e.get("foul_committed") or e.get("bad_behaviour") or {}).get("card", {})
        return e["type"]["name"] == "Substitution" or card.get("name") in RED_CARDS

    ordered = sorted(events, key=lambda e: e["index"])
    end = next((e for e in ordered if ends_window(e)), None)
    end_index = end["index"] if end else float("inf")
    end_minute = end["minute"] if end else max(e["minute"] for e in events)
    ix = {n: i for i, n in enumerate(names)}
    counts = np.zeros((len(names), len(names)), dtype=int)
    for e in completed_passes(ordered, team):
        if e["index"] >= end_index:
            break
        recipient = e["pass"].get("recipient")
        if recipient and e["player"]["name"] in ix and recipient["name"] in ix:
            counts[ix[e["player"]["name"]], ix[recipient["name"]]] += 1
    return names, positions, counts, max(int(end_minute), 1)


def window_stats(
    counts: np.ndarray, minutes: int, roles: list[str], lines: list[str]
) -> dict[str, float]:
    """Every window statistic for one pass-count matrix."""
    total = int(counts.sum())
    s = counts.sum(axis=0) + counts.sum(axis=1)
    active = s > 0
    n = int(active.sum())
    sym = counts + counts.T
    upper = np.triu_indices(len(counts), k=1)
    w = mk.transition_matrix(counts)
    passers = w.sum(axis=1) > 0
    row_entropy = -(w * np.log2(w, out=np.zeros_like(w), where=w > 0)).sum(axis=1)
    weights = counts.sum(axis=1) / total if total else np.zeros(len(counts))
    stats = {
        "passes": float(total),
        "intensity": total / minutes,
        "density": float((counts > 0).sum() / (n * (n - 1))) if n > 1 else 0.0,
        "centralization": float((s.max() - s[active]).sum() / ((n - 2) * total))
        if n > 2 and total
        else 0.0,
        "mean_entropy": float(row_entropy[passers].mean()) if passers.any() else 0.0,
        "reciprocity": float(2 * np.minimum(counts, counts.T)[upper].sum() / total)
        if total
        else 0.0,
        "top_pair_share": float(sym[upper].max() / total) if total else 0.0,
        "ei": mk.effective_information(w),
        "effectiveness": mk.effectiveness(w),
        "ce_roles": mk.causal_emergence(w, mk.labels_to_partition(roles), weights),
        "ce_lines": mk.causal_emergence(w, mk.labels_to_partition(lines), weights),
    }
    return stats


def resample_counts(counts: np.ndarray, rng: random.Random) -> np.ndarray:
    """Bootstrap: draw as many passes as observed, with replacement, from the observed passes."""
    pairs = np.argwhere(counts > 0)
    probs = counts[counts > 0] / counts.sum()
    draws = rng.choices(range(len(pairs)), weights=probs, k=int(counts.sum()))
    out = np.zeros_like(counts)
    for k, c in Counter(draws).items():
        out[tuple(pairs[k])] = c
    return out


def permutation_spearman(
    x: np.ndarray, y: np.ndarray, reps: int, rng: random.Random
) -> tuple[float, float]:
    """Spearman rho and a two-sided permutation p-value."""

    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
        for v in np.unique(a):  # average ranks for ties
            order[a == v] = order[a == v].mean()
        return order

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    ry_list = list(ry)
    hits = 0
    for _ in range(reps):
        rng.shuffle(ry_list)
        hits += abs(np.corrcoef(rx, ry_list)[0, 1]) >= abs(rho) - 1e-12
    return rho, (hits + 1) / (reps + 1)


# --------------------------------------------------------------- compute
def compute(
    matches: list[dict[str, Any]],
    events: dict[int, list[dict[str, Any]]],
    lineups: dict[int, list[dict[str, Any]]],
    config: dict[str, Any] = CONFIG,
) -> list[dict[str, Any]]:
    """Long-format result rows for every match plus season-level summaries."""
    team = config["team"]
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float | None, unit: str | None = None, **dims: Any) -> None:
        rows.append({"metric": metric, "value": value, "unit": unit, "dims": dims})

    boot_cfg, null_cfg, part_cfg = config["bootstrap"], config["null"], config["random_partitions"]
    season: dict[str, list[float]] = defaultdict(list)
    goal_diff: list[float] = []
    slot_counts: dict[tuple[str, str], int] = defaultdict(int)
    for m in sorted(matches, key=lambda m: m["match_date"]):
        match_id = m["match_id"]
        home = m["home_team"]["home_team_name"]
        at_home = home == team
        gf, ga = (
            (m["home_score"], m["away_score"]) if at_home else (m["away_score"], m["home_score"])
        )
        opponent = m["away_team"]["away_team_name"] if at_home else home
        result = "W" if gf > ga else "D" if gf == ga else "L"
        add("match_info", match_id, match=match_id, date=m["match_date"], home=home,
            away=m["away_team"]["away_team_name"], home_score=m["home_score"],
            away_score=m["away_score"], opponent=opponent, venue="H" if at_home else "A",
            result=result, featured=match_id in config["featured_matches"])  # fmt: skip
        goal_diff.append(gf - ga)

        # 1. Full-match network, exactly as the notebook.
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
        pos, where = average_positions(passes), main_positions(passes)
        names = labels(lineups[match_id], team)
        for player in g.nodes():
            x, y = pos[player]
            add("passes_made", made[player], "passes", match=match_id, player=player,
                label=names.get(player, player), position=where.get(player), x=x, y=y,
                received=received[player])  # fmt: skip
        for u, v, d in g.edges(data=True):
            add("pass_link", d["weight"], "passes", match=match_id, passer=u, recipient=v)

        # 2. Window network: stats, bootstrap CIs, configuration null.
        xi, positions, counts, minutes = window_counts(events[match_id], team)
        roles, lines = [role_of(p) for p in positions], [line_of(p) for p in positions]
        for i, (player, position) in enumerate(zip(xi, positions, strict=True)):
            add("window_player", int(counts[i].sum()), "passes", match=match_id, player=player,
                label=names.get(player, player), position=position, role=roles[i],
                line=lines[i], received=int(counts[:, i].sum()))  # fmt: skip
        for i, j in np.argwhere(counts > 0):
            slot_counts[(positions[i], positions[j])] += int(counts[i, j])
        add("window_minutes", minutes, "minutes", match=match_id)
        observed = window_stats(counts, minutes, roles, lines)
        for stat, value in observed.items():
            add(f"w_{stat}", value, match=match_id)
            season[stat].append(value)
        rng = random.Random(boot_cfg["seed"] + match_id)
        boot = [window_stats(resample_counts(counts, rng), minutes, roles, lines)
                for _ in range(boot_cfg["reps"])]  # fmt: skip
        for stat in WINDOW_STATS:
            # Basic (reverse-percentile) interval: entropy-based statistics are biased
            # upward in small samples and resampling exaggerates it, so the percentile
            # interval can exclude the estimate itself. bias = bootstrap mean - estimate.
            values = np.array([b[stat] for b in boot])
            q_lo, q_hi = np.percentile(values, [2.5, 97.5])
            add(f"w_{stat}_lo", float(2 * observed[stat] - q_hi), match=match_id)
            add(f"w_{stat}_hi", float(2 * observed[stat] - q_lo), match=match_id)
            add(f"w_{stat}_bias", float(values.mean() - observed[stat]), match=match_id)
        rng = random.Random(null_cfg["seed"] + match_id)
        null = [window_stats(mk.configuration_counts(counts, rng), minutes, roles, lines)
                for _ in range(null_cfg["reps"])]  # fmt: skip
        for stat in NULL_STATS:
            values = np.array([b[stat] for b in null])
            add(f"w_{stat}_null_mean", float(values.mean()), match=match_id)
            add(f"w_{stat}_null_sd", float(values.std()), match=match_id)
            add(f"w_{stat}_null_pct", float((values < observed[stat]).mean()), match=match_id)

        # 3. Markov chain: role partition vs random partitions; greedy search.
        w = mk.transition_matrix(counts)
        weights = counts.sum(axis=1) / counts.sum()
        role_partition = mk.labels_to_partition(roles)
        rng = random.Random(part_cfg["seed"] + match_id)
        random_ce = [mk.causal_emergence(w, mk.random_partition_like(role_partition, len(w), rng),
                                         weights) for _ in range(part_cfg["reps"])]  # fmt: skip
        add("w_ce_roles_random_mean", float(np.mean(random_ce)), match=match_id)
        add("w_ce_roles_vs_random_pct", float(np.mean(np.array(random_ce) < observed["ce_roles"])),
            match=match_id)  # fmt: skip
        season["ce_roles_vs_random_pct"].append(
            float(np.mean(np.array(random_ce) < observed["ce_roles"]))
        )
        best, best_ei = mk.greedy_partition(w, weights)
        add("w_ce_greedy", best_ei - observed["ei"], match=match_id)
        for k, group in enumerate(g for g in best if len(g) > 1):
            add("greedy_group", len(group), match=match_id, group=k,
                players=[names.get(xi[i], xi[i]) for i in group])  # fmt: skip
        if match_id in config["featured_matches"]:
            macro = mk.macro_matrix(w, role_partition, weights)
            role_names = [roles[g[0]] for g in role_partition]
            for a, src in enumerate(role_names):
                for b, dst in enumerate(role_names):
                    add(
                        "role_transition",
                        float(macro[a, b]),
                        match=match_id,
                        source=src,
                        target=dst,
                    )

    # Season summaries.
    n_matches = len(goal_diff)
    for stat, values in season.items():
        add(f"season_{stat}_mean", float(np.mean(values)))
    add("season_matches", n_matches, "matches")
    add("season_ce_roles_negative", int(sum(v < 0 for v in season["ce_roles"])), "matches")
    add("season_ce_lines_negative", int(sum(v < 0 for v in season["ce_lines"])), "matches")
    add("season_ce_roles_beats_random_median",
        int(sum(v > 0.5 for v in season["ce_roles_vs_random_pct"])), "matches")  # fmt: skip
    test_cfg = config["season_tests"]
    rng = random.Random(test_cfg["seed"])
    for stat in ("intensity", "centralization", "ei") if n_matches >= 3 else ():
        rho, p = permutation_spearman(np.array(season[stat]), np.array(goal_diff),
                                      test_cfg["permutations"], rng)  # fmt: skip
        add("season_spearman_goal_diff", rho, stat=stat)
        add("season_spearman_goal_diff_p", p, stat=stat)

    # Season pooled by starting-position slot (much larger sample than one match).
    slots = sorted({a for a, _ in slot_counts} | {b for _, b in slot_counts})
    ix = {s: i for i, s in enumerate(slots)}
    pooled = np.zeros((len(slots), len(slots)), dtype=int)
    for (a, b), c in slot_counts.items():
        pooled[ix[a], ix[b]] = c
    w = mk.transition_matrix(pooled)
    weights = pooled.sum(axis=1) / pooled.sum()
    add("pooled_passes", int(pooled.sum()), "passes")
    add("pooled_slots", len(slots), "slots")
    add("pooled_ei", mk.effective_information(w), "bits")
    add("pooled_effectiveness", mk.effectiveness(w))
    for name, f in (("roles", role_of), ("lines", line_of)):
        add(
            f"pooled_ce_{name}",
            mk.causal_emergence(w, mk.labels_to_partition([f(s) for s in slots]), weights),
            "bits",
        )
    best, best_ei = mk.greedy_partition(w, weights)
    add("pooled_ce_greedy", best_ei - mk.effective_information(w), "bits")
    for k, group in enumerate(g for g in best if len(g) > 1):
        add("pooled_greedy_group", len(group), group=k, slots=[slots[i] for i in group])
    for s in slots:
        add("pooled_slot_passes", int(pooled[ix[s]].sum()), "passes", slot=s, role=role_of(s))
    return rows


# --------------------------------------------------------------- run
def _download(client: Client, dataset: dict[str, Any]) -> bytes:
    bucket, path = dataset["storage_path"].split("/", 1)
    data = gzip.decompress(client.storage.from_(bucket).download(path))
    if sha256_bytes(data) != dataset["sha256"]:
        raise RuntimeError(f"{path} does not match its recorded sha256")
    return data


def load_season(client: Client, datasets: dict[str, dict[str, Any]]) -> tuple[list, dict, dict]:
    """Matches list and ``{match_id: records}`` for events and lineups, sha256-verified."""
    matches = json.loads(_download(client, datasets[MATCHES_SLUG]))
    tables = []
    for kind in ("events", "lineups"):
        files = read_bundle(_download(client, datasets[season_slug(kind)]))
        tables.append({int(name.split("/")[-1].removesuffix(".json")): json.loads(b)
                       for name, b in files.items()})  # fmt: skip
    return matches, tables[0], tables[1]


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
            raise RuntimeError(f"Dataset {slug} missing; run pipelines.ingest.statsbomb --season")
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
        matches, events, lineups = load_season(client, datasets)
        write_results(client, run_id, compute(matches, events, lineups, CONFIG))
    return run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-dirty", action="store_true", help="record a run from uncommitted code"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_id = run(get_client(), allow_dirty=args.allow_dirty)
    log.info(
        "skipped: identical run already succeeded" if run_id is None else f"run {run_id} succeeded"
    )


if __name__ == "__main__":
    main()
