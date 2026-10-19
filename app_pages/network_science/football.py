import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import db
from lab.registry import entry
from lab.ui import empty_state, page_header, provenance_footer

ENTRY = entry("arsenal-passing-networks")
CODE_PATHS = (
    "pipelines/ingest/statsbomb.py",
    "pipelines/common/markov.py",
    "pipelines/experiments/network_science/football.py",
)
# Categorical slots 1-2 (validated palette), assigned in fixed order; gray for references.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
RESULT_COLORS = {"W": BLUE, "D": ORANGE, "L": GRAY}
RESULT_NAMES = {"W": "Win", "D": "Draw", "L": "Loss"}
PITCH, LINES = "#2d8c3c", "rgba(255, 255, 255, 0.85)"  # the pitch is a surface, not data
LENGTH, WIDTH = 120, 80  # StatsBomb pitch coordinates; y runs top to bottom
# StatsBomb's licence requires its logo on any published analysis of the data.
ASSETS = Path(__file__).resolve().parents[2] / "lab" / "assets"
SEASON_METRICS = {
    "intensity": ("Intensity", "completed passes per minute"),
    "centralization": ("Centralization", "0 = everyone equally involved, 1 = one hub"),
    "density": ("Density", "share of possible passing links used"),
    "mean_entropy": ("Passing entropy", "bits; how evenly players spread their passes"),
    "reciprocity": ("Reciprocity", "share of passes returned along the same link"),
    "ei": ("Effective information", "bits; how specific the passing chain is"),
}
NULL_METRICS = {
    "ei": "Effective information",
    "mean_entropy": "Passing entropy",
    "density": "Density",
    "top_pair_share": "Strongest partnership",
    "reciprocity": "Reciprocity",
    "ce_roles": "Causal emergence (roles)",
}

page_header(ENTRY)

st.header("Purpose")
st.write(
    "A football team is a network: players are nodes and every completed pass is a directed, "
    "weighted edge. I started by drawing two matches from Arsène Wenger's 2003/04 "
    "*Invincibles*, a 2-1 home win over Chelsea and the 0-0 draw at Old Trafford, to see who "
    "the ball flowed through and which partnerships carried it. Here I extend that to all 38 "
    "matches of the unbeaten league season and ask two harder questions: is Arsenal's passing "
    "structure more than what pass volumes alone would produce, and is the team better "
    "described as eleven individual players or as a handful of positional roles? Finally I "
    "follow each possession to its end (a shot, a lost ball or a restart) to ask whether "
    "*where* the ball is matters more than *who* has it."
)

st.header("Method")
st.markdown(
    "- **Data:** [StatsBomb open data](https://github.com/statsbomb/open-data) event files for "
    "all 38 Premier League matches of 2003/04 (Arsenal's matches are the only ones in the "
    "set). Lineups supply players' common names (Lauren, Edu, Gilberto Silva).\n"
    "- **Full-match network** (the pitch view): Arsenal's *completed* passes; an edge "
    "*A → B* is weighted by how many passes A completed to B; each player sits at the "
    "average of where they made and received passes. Substitutes appear if they touched "
    "the ball.\n"
    "- **Window network** (everything else): only the starting XI, from kick-off until "
    "Arsenal's first substitution or red card, so that every match compares the same eleven "
    "nodes. Median window: about 69 minutes.\n"
    "- **Structure:** intensity (passes per minute), density, centralization of involvement, "
    "each player's passing entropy, reciprocity and the strongest partnership's share. "
    "Definitions are in the run config below.\n"
    "- **Uncertainty:** 500 bootstrap resamples of each match's passes, with *basic* 95% "
    "intervals. Entropy-based measures are biased upward in samples this small, and the "
    "basic interval corrects for that bias.\n"
    "- **Against chance:** 500 random versions of each match that keep every player's numbers "
    "of passes made and received, but reassign who passed to whom (a configuration model). "
    "A measure is *beyond chance* when it falls outside the middle 95% of those versions.\n"
    "- **Players or positions:** each player's passes, as shares, define a Markov chain of "
    "possession. Its *effective information* (EI; Klein & Hoel 2020) measures how specifically "
    "the ball's next destination depends on who has it. *Causal emergence* is the EI gained "
    "by merging players into larger units, either roles (side × line, so the centre-back "
    "pair is one unit) or lines (keeper, defence, midfield, attack). I compare roles with "
    "500 random groupings of the same sizes, and let a greedy search look for any grouping "
    "that helps. Finally I pool the whole season by starting position, which gives about "
    "11,700 passes and removes most small-sample bias.\n"
    "- **Possession chains:** the same chain, but following each Arsenal possession to its "
    "end: **Shot**, **Lost** (the opponent has the next possession) or **Restart** (Arsenal "
    "keep it after a stoppage). End states are absorbing. Because absorbing states are "
    "perfectly deterministic, they can inflate EI, so every grouping is compared with "
    "random groupings of the same sizes and with the trivial grouping of all players into "
    "one, and every result is repeated with the end states as sinks. Pooled over the season, "
    "the micro states are *starting position × third of the pitch* where the ball was "
    "received, so I can merge by *who* (position), by *where* (third), or by both."
)

run = db.latest_run(ENTRY.slug)
results = db.results(run.id) if run else pd.DataFrame()
if run is None or results.empty:
    st.header("Results")
    empty_state("No published results yet. They appear here once the pipeline has run.")
    st.stop()

scalars = results[results["dims"].map(len) == 0].set_index("metric")["value"]


def value(metric: str) -> float:
    """A season-level scalar, or NaN if this run did not record it."""
    v = scalars.get(metric)
    return float("nan") if v is None or pd.isna(v) else float(v)


def series(metric: str, *dims: str) -> pd.DataFrame:
    """Rows of ``metric`` with the requested dims expanded into columns."""
    rows = results[results["metric"] == metric]
    out = pd.DataFrame({d: rows["dims"].map(lambda x, d=d: x.get(d)) for d in dims})
    out["value"] = rows["value"].to_numpy()
    return out.reset_index(drop=True)


matches = series("match_info", "match", "date", "home", "away", "home_score", "away_score",
                 "opponent", "venue", "result", "featured")  # fmt: skip
if matches.empty:
    st.header("Results")
    empty_state("This run recorded no matches.")
    st.stop()
matches["match"] = matches["match"].astype(int)
matches = matches.sort_values("date").reset_index(drop=True)
matches["title"] = matches.apply(
    lambda m: f"{m.home} {m.home_score}-{m.away_score} {m.away}", axis=1
)
titles = dict(zip(matches["match"], matches["title"], strict=True))
per_match = (
    results[results["dims"].map(lambda d: set(d) == {"match"})]
    .assign(match=lambda r: r["dims"].map(lambda d: int(d["match"])))
    .pivot_table(index="match", columns="metric", values="value", aggfunc="first")
)
season = matches.set_index("match").join(per_match)
players = series("passes_made", "match", "player", "label", "position", "x", "y", "received")
links = series("pass_link", "match", "passer", "recipient")


def stat(match_id: int, metric: str) -> float:
    """A per-match scalar, or NaN if this run did not record it."""
    try:
        return float(season.at[match_id, metric])
    except KeyError:
        return float("nan")


def beyond(metric: str) -> tuple[int, int, int]:
    """Matches where the observed value is below, inside or above the null's middle 95%."""
    pct = season.get(f"w_{metric}_null_pct", pd.Series(dtype=float)).dropna()
    return (
        int((pct < 0.025).sum()),
        int(((pct >= 0.025) & (pct <= 0.975)).sum()),
        int((pct > 0.975).sum()),
    )


def pitch_shapes() -> list[dict]:
    """Pitch markings in StatsBomb coordinates."""
    line = {"color": LINES, "width": 1.5}

    def rect(x0: float, y0: float, x1: float, y1: float) -> dict:
        return {"type": "rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "line": line}

    def circle(cx: float, cy: float, r: float) -> dict:
        return {"type": "circle", "x0": cx - r, "y0": cy - r, "x1": cx + r, "y1": cy + r} | {
            "line": line
        }

    shapes = [
        rect(0, 0, LENGTH, WIDTH),
        {"type": "line", "x0": 60, "y0": 0, "x1": 60, "y1": WIDTH, "line": line},
        circle(60, 40, 10),
    ]
    for goal_x, sign in ((0, 1), (LENGTH, -1)):
        shapes += [
            rect(goal_x, 18, goal_x + sign * 18, 62),
            rect(goal_x, 30, goal_x + sign * 6, 50),
            rect(goal_x, 36, goal_x - sign * 2, 44),
        ]
    return shapes


def network_figure(match_id: int, min_passes: int) -> go.Figure:
    """Players at their average positions, links drawn with width by pass count."""
    nodes = players[players["match"] == match_id].set_index("player")
    edges = links[(links["match"] == match_id) & (links["value"] >= min_passes)]
    heaviest = float(edges["value"].max()) if not edges.empty else 1.0
    fig = go.Figure()
    for e in edges.sort_values("value").itertuples():
        if e.passer not in nodes.index or e.recipient not in nodes.index:
            continue
        x0, y0 = nodes.at[e.passer, "x"], nodes.at[e.passer, "y"]
        x1, y1 = nodes.at[e.recipient, "x"], nodes.at[e.recipient, "y"]
        # Shift each direction 0.8 m to its own side so A→B and B→A don't overlap, and
        # stop 2.5 m short of both players so links don't run through the markers.
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        ox, oy = -uy * 0.8, ux * 0.8
        trim = min(2.5, length / 3)
        x0, y0, x1, y1 = x0 + ux * trim, y0 + uy * trim, x1 - ux * trim, y1 - uy * trim
        share = e.value / heaviest
        color = f"rgba(255, 255, 255, {0.25 + 0.6 * share:.2f})"
        hover = (
            f"{nodes.at[e.passer, 'label']} → {nodes.at[e.recipient, 'label']}<br>"
            f"{e.value:.0f} passes<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[x0 + ox, x1 + ox],
                y=[y0 + oy, y1 + oy],
                mode="lines",
                line={"color": color, "width": 1 + 7 * share},
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x0 + ox + 0.75 * (x1 - x0)],
                y=[y0 + oy + 0.75 * (y1 - y0)],
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 6 + 6 * share,
                    "angle": math.degrees(math.atan2(x1 - x0, y0 - y1)),  # y axis is reversed
                    "color": color,
                },
                hovertemplate=hover,
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=nodes["x"],
            y=nodes["y"],
            mode="markers+text",
            marker={
                "size": 10 + nodes["value"].clip(lower=1) ** 0.5 * 3.5,
                "color": BLUE,
                "line": {"color": "white", "width": 2},
            },
            text=[
                label if label != player else label.split()[-1]
                for player, label in zip(nodes.index, nodes["label"], strict=True)
            ],
            textposition="bottom center",
            textfont={"color": "white", "size": 12, "shadow": "1px 1px 2px #000"},
            customdata=nodes[["label", "position", "value", "received"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                "%{customdata[2]:.0f} passes made · %{customdata[3]:.0f} received<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        shapes=pitch_shapes(),
        plot_bgcolor=PITCH,
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        hovermode="closest",
    )
    fig.update_xaxes(range=[-3, LENGTH + 3], visible=False, fixedrange=True)
    fig.update_yaxes(
        range=[WIDTH + 3, -3], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1
    )
    return fig


st.header("Results")
n_matches = len(matches)
record = matches["result"].value_counts()
with st.container(horizontal=True):
    st.metric("Matches", f"{n_matches}", border=True)
    st.metric("Record", f"{record.get('W', 0)}W {record.get('D', 0)}D {record.get('L', 0)}L",
              border=True)  # fmt: skip
    st.metric("Window passes per match", f"{season['w_passes'].mean():.0f}", border=True)
    st.metric("Pooled passes", f"{value('pooled_passes'):,.0f}", border=True)

overview, pitch, chance, scales, possession = st.tabs(
    [
        "Season overview",
        "Match on the pitch",
        "Against chance",
        "Players or positions?",
        "Possession chains",
    ]
)

with overview:
    choice = st.segmented_control(
        "Measure",
        options=list(SEASON_METRICS),
        format_func=lambda k: SEASON_METRICS[k][0],
        default="intensity",
        key="season_metric",
    )
    choice = choice or "intensity"
    label, help_text = SEASON_METRICS[choice]
    fig = go.Figure()
    for result, color in RESULT_COLORS.items():
        part = season[season["result"] == result]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(part["date"]),
                y=part[f"w_{choice}"],
                mode="markers",
                name=RESULT_NAMES[result],
                marker={"size": 10, "color": color, "line": {"color": "white", "width": 2}},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (part[f"w_{choice}_hi"] - part[f"w_{choice}"]).clip(lower=0),
                    "arrayminus": (part[f"w_{choice}"] - part[f"w_{choice}_lo"]).clip(lower=0),
                    "color": color,
                    "thickness": 1.5,
                    "width": 0,
                },
                customdata=part[["title", "venue"]].to_numpy(),
                hovertemplate=(
                    f"<b>%{{customdata[0]}}</b> (%{{customdata[1]}})<br>{label}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        yaxis_title=f"{label} ({help_text})",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(fig, key="season_chart", width="stretch", theme="streamlit")
    st.caption(
        "One point per match (starting-XI window), with its 95% bootstrap interval. For "
        "entropy and effective information the interval sits below the point: the plug-in "
        "estimate is biased upward, and the interval corrects for it."
    )
    rho = series("season_spearman_goal_diff", "stat").set_index("stat")["value"]
    pval = series("season_spearman_goal_diff_p", "stat").set_index("stat")["value"]
    if not rho.empty:
        st.markdown(
            "**Does structure track the scoreline?** Spearman correlation with goal "
            "difference, with permutation p-values: "
            + "; ".join(
                f"{SEASON_METRICS[s][0].lower()} ρ = {rho[s]:+.2f} (p = {pval[s]:.2f})"
                for s in rho.index
                if s in SEASON_METRICS
            )
            + ". None is distinguishable from chance. With 38 matches and no defeats I "
            "treat this as exploratory."
        )

with pitch:
    featured = matches[matches["featured"].astype(bool)]["match"].tolist()
    default = featured[0] if featured else int(matches["match"].iloc[0])
    with st.container(horizontal=True, vertical_alignment="bottom"):
        match_id = st.selectbox(
            "Match",
            options=matches["match"].tolist(),
            index=matches["match"].tolist().index(default),
            format_func=lambda m: f"{season.at[m, 'date']} · {titles[m]}",
            key="match",
        )
        min_passes = st.slider("Draw links with at least", 1, 10, 3, key="min_passes")
    with st.container(horizontal=True):
        st.metric("Completed passes", f"{stat(match_id, 'completed_passes'):,.0f}", border=True)
        st.metric("Players", f"{stat(match_id, 'players'):.0f}", border=True)
        st.metric("Connections", f"{stat(match_id, 'connections'):.0f}", border=True)
        st.metric("Avg degree", f"{stat(match_id, 'average_degree'):.2f}", border=True)
    st.plotly_chart(network_figure(match_id, min_passes), key="network", width="stretch")
    st.caption(
        "Full match. Hover a player or a link for counts. Line width and opacity scale with "
        "passes on that link; arrows point from passer to recipient."
    )
    nodes = players[players["match"] == match_id]
    left, right = st.columns(2)
    with left:
        st.markdown("**Strongest links**")
        names = dict(zip(nodes["player"], nodes["label"], strict=True))
        top = links[links["match"] == match_id].nlargest(10, "value")
        st.dataframe(
            pd.DataFrame(
                {
                    "Passer": top["passer"].map(names),
                    "Recipient": top["recipient"].map(names),
                    "Passes": top["value"].astype(int),
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**Passes made, by player**")
        ranked = nodes.sort_values("value")
        bars = go.Figure(
            go.Bar(
                x=ranked["value"],
                y=ranked["label"],
                orientation="h",
                marker_color=BLUE,
                customdata=ranked[["position", "received"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b> (%{customdata[0]})<br>%{x:.0f} made · "
                    "%{customdata[1]:.0f} received<extra></extra>"
                ),
            )
        )
        bars.update_layout(
            height=380, margin={"l": 0, "r": 0, "t": 10, "b": 0}, xaxis_title="Completed passes"
        )
        st.plotly_chart(bars, key="passers", width="stretch", theme="streamlit")

with chance:
    counts = {k: beyond(k) for k in NULL_METRICS}
    st.dataframe(
        pd.DataFrame(
            {
                "Measure": [NULL_METRICS[k] for k in counts],
                "Below chance": [c[0] for c in counts.values()],
                "Within chance": [c[1] for c in counts.values()],
                "Above chance": [c[2] for c in counts.values()],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Number of matches (of {n_matches}) where Arsenal's value falls below, inside or "
        "above the middle 95% of 500 random versions that keep each player's pass volumes."
    )
    pick = st.segmented_control(
        "Show",
        options=list(NULL_METRICS),
        format_func=NULL_METRICS.get,
        default="ei",
        key="null_metric",
    )
    pick = pick or "ei"
    ordered = season.sort_values("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(ordered["date"]),
            y=ordered[f"w_{pick}_null_mean"],
            mode="markers",
            name="Random versions (mean ± 2 sd)",
            marker={"size": 8, "color": GRAY, "symbol": "line-ew-open", "line": {"width": 2}},
            error_y={
                "type": "data",
                "array": 2 * ordered[f"w_{pick}_null_sd"],
                "color": GRAY,
                "thickness": 6,
                "width": 0,
            },
            hovertemplate="Random: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(ordered["date"]),
            y=ordered[f"w_{pick}"],
            mode="markers",
            name="Arsenal",
            marker={"size": 9, "color": BLUE, "line": {"color": "white", "width": 2}},
            customdata=ordered[["title"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>Arsenal: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        yaxis_title=NULL_METRICS[pick],
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(fig, key="null_chart", width="stretch", theme="streamlit")

with scales:
    roles_neg = value("season_ce_roles_negative")
    lines_neg = value("season_ce_lines_negative")
    roles_upper_neg = int((season["w_ce_roles_hi"] < 0).sum())
    with st.container(horizontal=True):
        st.metric("Mean EI, players", f"{season['w_ei'].mean():.2f} bits", border=True)
        st.metric("Grouping by role loses information", f"{roles_neg:.0f} / {n_matches}",
                  border=True)  # fmt: skip
        st.metric("Grouping by line loses information", f"{lines_neg:.0f} / {n_matches}",
                  border=True)  # fmt: skip
        st.metric("Pooled season, roles", f"{value('pooled_ce_roles'):+.2f} bits", border=True)
    ordered = season.sort_values("date")
    fig = go.Figure()
    fig.add_hline(y=0, line={"color": GRAY, "width": 1, "dash": "dash"})
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(ordered["date"]),
            y=ordered["w_ce_roles_random_mean"],
            mode="markers",
            name="Random groupings, same sizes (mean)",
            marker={"size": 9, "color": GRAY, "symbol": "diamond-open", "line": {"width": 2}},
            hovertemplate="Random groupings: %{y:+.3f} bits<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(ordered["date"]),
            y=ordered["w_ce_roles"],
            mode="markers",
            name="Grouping by role",
            marker={"size": 9, "color": BLUE, "line": {"color": "white", "width": 2}},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (ordered["w_ce_roles_hi"] - ordered["w_ce_roles"]).clip(lower=0),
                "arrayminus": (ordered["w_ce_roles"] - ordered["w_ce_roles_lo"]).clip(lower=0),
                "color": BLUE,
                "thickness": 1.5,
                "width": 0,
            },
            customdata=ordered[["title"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>Roles: %{y:+.3f} bits<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        yaxis_title="Causal emergence (bits; above 0 = grouping helps)",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(fig, key="ce_chart", width="stretch", theme="streamlit")
    st.caption(
        f"Below the dashed line, describing the team by roles loses information compared "
        f"with describing it by players. The whole 95% interval is below zero in "
        f"{roles_upper_neg} of {n_matches} matches. Intervals sit mostly above their points "
        "because player-level EI is inflated in small samples, which pushes this estimate "
        "down; the interval corrects for that."
    )
    transitions = series("role_transition", "match", "source", "target")
    focus = transitions[transitions["match"] == default]
    if not focus.empty:
        order = ["GK", "L-DEF", "C-DEF", "R-DEF", "L-MID", "C-MID", "R-MID", "L-ATT", "C-ATT",
                 "R-ATT"]  # fmt: skip
        present = [r for r in order if r in set(focus["source"])] + sorted(
            set(focus["source"]) - set(order)
        )
        grid = focus.pivot(index="source", columns="target", values="value").reindex(
            index=present, columns=present
        )
        heat = go.Figure(
            go.Heatmap(
                z=grid.to_numpy(),
                x=present,
                y=present,
                colorscale=[[0, "#f0efec"], [1, "#104281"]],
                zmin=0,
                zmax=float(grid.max().max()),
                colorbar={"title": "P(next)"},
                hovertemplate="%{y} → %{x}: %{z:.0%}<extra></extra>",
            )
        )
        heat.update_layout(
            height=420,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            xaxis_title="Ball goes to",
            yaxis_title="Ball is with",
            yaxis={"autorange": "reversed"},
            title=f"The role-level chain, {titles[default]}",
        )
        st.plotly_chart(heat, key="role_chain", width="stretch", theme="streamlit")
        st.caption(
            "What the coarse-grained description looks like: transition probabilities between "
            "roles, weighted by how often each player passes. The diagonal is passing within "
            "a role (for example, centre-back to centre-back)."
        )
    groups = series("pooled_greedy_group", "group", "slots")
    found = "; ".join(" + ".join(g) for g in groups["slots"]) if not groups.empty else ""
    st.markdown(
        f"**Pooled season.** By starting position over the season "
        f"({value('pooled_passes'):,.0f} passes, {value('pooled_slots'):.0f} slots), grouping "
        f"by role changes EI by {value('pooled_ce_roles'):+.2f} bits and by line "
        f"{value('pooled_ce_lines'):+.2f}. The best grouping the greedy search found gains "
        f"{value('pooled_ce_greedy'):+.3f} bits"
        + (f", by merging {found}" if found else "")
        + ". That is negligible."
    )


SCALE_NAMES = {
    "who": "Who: starting position",
    "where": "Where: third of the pitch",
    "role_third": "Role × third",
    "line_third": "Line × third",
    "role": "Role",
    "line": "Line",
    "one_group": "All players merged (artefact baseline)",
}
pooled_scales = series("pp_scale", "scale", "groups", "ce_sinks", "random_mean", "vs_random_pct",
                       "lo", "hi").set_index("scale")  # fmt: skip

with possession:
    if "p_transitions" not in season or pooled_scales.empty:
        empty_state("This run has no possession chains; it predates them.")
    else:
        with st.container(horizontal=True):
            st.metric("Shots per match", f"{season['p_shot'].mean():.1f}", border=True)
            st.metric("Possessions lost per match", f"{season['p_lost'].mean():.1f}", border=True)
            st.metric("Restarts per match", f"{season['p_restart'].mean():.1f}", border=True)
            st.metric("Pooled transitions", f"{value('pp_transitions'):,.0f}", border=True)
        ordered = season.sort_values("date")
        fig = go.Figure()
        fig.add_hline(y=0, line={"color": GRAY, "width": 1, "dash": "dash"})
        for column, name, color, symbol in (
            ("p_one_group_ce", "All players merged into one", GRAY, "x-thin-open"),
            ("p_lines_random_mean", "Random groupings, same sizes as lines", GRAY, "diamond-open"),
            ("p_lines_ce", "Grouping by line", BLUE, "circle"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(ordered["date"]),
                    y=ordered[column],
                    mode="markers",
                    name=name,
                    marker={"size": 9, "color": color, "symbol": symbol, "line": {"width": 2}},
                    customdata=ordered[["title"]].to_numpy(),
                    hovertemplate=f"<b>%{{customdata[0]}}</b><br>{name}: %{{y:+.3f}} bits"
                    "<extra></extra>",
                )
            )
        fig.update_layout(
            height=380,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            yaxis_title="Causal emergence (bits), end states absorbing",
            legend={"orientation": "h", "y": 1.14},
        )
        st.plotly_chart(fig, key="possession_chart", width="stretch", theme="streamlit")
        st.caption(
            f"Once possessions can end, grouping players by line *gains* information in "
            f"{value('season_p_lines_ce_positive'):.0f} of {n_matches} matches. But random "
            f"groupings of the same sizes gain just as much, lines beat the typical random "
            f"grouping in only {value('season_p_lines_beats_random_median'):.0f} matches, and "
            f"merging every player into one gains the most of all. With the end states treated "
            f"as sinks instead, grouping by line gains information in "
            f"{value('season_p_lines_ce_sinks_positive'):.0f} matches and merging everyone in "
            f"{value('season_p_one_group_ce_sinks_positive'):.0f}."
        )
        table = pooled_scales.reindex([k for k in SCALE_NAMES if k in pooled_scales.index])
        st.markdown("**Season pooled: which scale keeps the most information?**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Scale": [SCALE_NAMES[k] for k in table.index],
                    "Groups": table["groups"].astype(int),
                    "Causal emergence (bits)": table["value"].map("{:+.3f}".format),
                    "Bias-corrected 95% interval": [
                        "–" if pd.isna(lo) else f"{lo:+.2f} to {hi:+.2f}"
                        for lo, hi in zip(table["lo"], table["hi"], strict=True)
                    ],
                    "Random, same sizes": table["random_mean"].map("{:+.3f}".format),
                    "Beats random": [
                        "–" if pd.isna(v) else f"{v:.0%}" for v in table["vs_random_pct"]
                    ],
                    "End states as sinks": table["ce_sinks"].map("{:+.3f}".format),
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Micro scale: {value('pp_states'):.0f} states (starting position × third where "
            f"the ball was received) plus the three end states, from "
            f"{value('pp_transitions'):,.0f} transitions. *Beats random* is the share of "
            "random groupings of the same sizes that the scale outperforms."
        )

st.header("Takeaways")
ei_below, _, ei_above = beyond("ei")
ent_below, _, _ = beyond("mean_entropy")
dens_below, _, _ = beyond("density")
_, rec_within, _ = beyond("reciprocity")
ce_below, _, _ = beyond("ce_roles")
beats_random = value("season_ce_roles_beats_random_median")
bullets = []
for m in matches[matches["featured"].astype(bool)].itertuples():
    mp = players[players["match"] == m.match]
    ml = links[links["match"] == m.match]
    if mp.empty or ml.empty:
        continue
    names = dict(zip(mp["player"], mp["label"], strict=True))
    hub = mp.loc[mp["value"].idxmax()]
    top = ml.loc[ml["value"].idxmax()]
    back = ml[(ml["passer"] == top.recipient) & (ml["recipient"] == top.passer)]["value"]
    pair = float(top.value) + (float(back.iloc[0]) if not back.empty else 0.0)
    bullets.append(
        f"**{m.title}.** {hub.label} ({hub.position}) made the most completed passes "
        f"({hub.value:.0f} of {stat(m.match, 'completed_passes'):.0f}), and the busiest pair, "
        f"{names[top.passer]} and {names[top.recipient]}, exchanged {pair:.0f} "
        f"({pair / stat(m.match, 'completed_passes'):.0%})."
    )
bullets += [
    f"**Specific, not just busy.** Given how many passes each player made and received, "
    f"Arsenal's passing was more specific than chance in {ei_above} of {n_matches} matches "
    f"(effective information above the random versions), with fewer distinct links "
    f"({dens_below} matches) and more targeted individual passing ({ent_below} matches). "
    f"Reciprocity, by contrast, was what chance predicts in {rec_within} matches. I read "
    "this as players having preferred partners, without that meaning give-and-go play. "
    "That specificity is determinism (each player's passes are focused), not low "
    "redundancy: Arsenal's degeneracy is about what chance predicts (see the causal "
    "emergence entry).",
    f"**Eleven players, not four lines.** Merging players into roles lost information in "
    f"{roles_neg:.0f} of {n_matches} matches, and merging into lines lost it in "
    f"{lines_neg:.0f}. Pooling the whole season, which removes most small-sample bias, "
    f"gives the same answer ({value('pooled_ce_roles'):+.2f} bits for roles). Roles were "
    f"still the least-bad grouping: they beat the typical random grouping of the same sizes "
    f"in {beats_random:.0f} matches. In Klein and Hoel's terms there is no causal emergence "
    "at the positional scale.",
    f"**Why roles lose more than chance would.** Grouping by role cost more information "
    f"than it would in a volume-matched random team in {ce_below} of {n_matches} matches. "
    "That is the flip side of the previous point: the information lives in who each "
    "player passes to, and merging, say, the two centre-backs blurs two distinct passing "
    "profiles into one. I think that is a real statement about this team, but it rests on "
    "completed passes only.",
]
if not pooled_scales.empty and {"where", "line", "one_group"} <= set(pooled_scales.index):
    where, line_, blob = (pooled_scales.loc[k] for k in ("where", "line", "one_group"))
    bullets += [
        f"**Following possessions to their end creates emergence, but only on paper.** With "
        f"shots, lost balls and restarts as absorbing end states, merging players into lines "
        f"gains information in {value('season_p_lines_ce_positive'):.0f} of {n_matches} "
        "matches. Random groupings of the same sizes gain as much, and merging everyone into "
        "a single blob gains the most. The deterministic end states take up a bigger share of "
        "a smaller chain, which raises EI with no football structure involved. I think this "
        "is the most useful methodological point here: causal emergence needs to be "
        "benchmarked against random and trivial groupings, not just against zero.",
        f"**Pitch thirds beat positional lines, at the same coarseness.** Pooled over the "
        f"season, merging states into three thirds of the pitch loses less information than "
        f"merging them into four positional lines, whichever way the end states are treated "
        f"({where['value']:+.2f} vs {line_['value']:+.2f} bits absorbing, "
        f"{where['ce_sinks']:+.2f} vs {line_['ce_sinks']:+.2f} as sinks). Thirds also beat "
        f"random groupings of their size ({where['random_mean']:+.2f}), while lines do no "
        f"better than random ({line_['random_mean']:+.2f}). No scale clears the all-in-one "
        f"baseline ({blob['lo']:+.2f} to {blob['hi']:+.2f} bits after bias correction), so "
        "there is still no genuine causal emergence. My reading is that where the ball is "
        "tells you more about what happens next than which line the player belongs to. "
        "Which scale looks best overall depends on how the end states are treated, and I "
        "think that sensitivity is itself worth reporting.",
    ]
st.markdown("\n".join(f"- {b}" for b in bullets))
follow_up = entry("causal-emergence-worm-vs-email")
st.page_link(
    follow_up.path,
    label="Follow-up: Arsenal next to the worm and Enron, in causal-emergence terms",
    icon=follow_up.icon,
)

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
theme = "dark" if st.context.theme.type == "dark" else "light"
st.image(ASSETS / f"statsbomb-logo-{theme}.png", width=220)
st.caption("Data: StatsBomb open data. Analysis and opinions are my own, not StatsBomb's.")
