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
    "pipelines/experiments/network_science/football.py",
)
BLUE = "#2a78d6"  # categorical slot 1 (validated palette)
PITCH, LINES = "#2d8c3c", "rgba(255, 255, 255, 0.85)"  # the pitch is a surface, not data
LENGTH, WIDTH = 120, 80  # StatsBomb pitch coordinates; y runs top to bottom
# StatsBomb's licence requires its logo on any published analysis of the data.
ASSETS = Path(__file__).resolve().parents[2] / "lab" / "assets"

page_header(ENTRY)

st.header("Purpose")
st.write(
    "A football team is a network: players are nodes and every completed pass is a directed, "
    "weighted edge. I wanted to see what Arsène Wenger's 2003/04 *Invincibles* look like "
    "drawn that way: who the ball flowed through, which partnerships carried it, and where "
    "on the pitch each player actually did their work. I compare two matches from that "
    "unbeaten season, a 2-1 home win over Chelsea and the 0-0 draw at Old Trafford."
)

st.header("Method")
st.markdown(
    "- **Data:** [StatsBomb open data](https://github.com/statsbomb/open-data) event files for "
    "both matches: every pass with its passer, recipient, start and end location and outcome. "
    "Lineups supply players' common names (Lauren, Edu, Gilberto Silva).\n"
    "- **Network:** Arsenal's *completed* passes only (StatsBomb records no outcome for a "
    "completed pass). An edge *A → B* is weighted by how many passes A completed to B. "
    "Substitutes appear if they touched the ball.\n"
    "- **Position on the pitch:** each player sits at the average of where they made their "
    "completed passes and where they received them. Node size is passes made.\n"
    "- **Drawing:** to keep the picture readable, only links of 3 or more passes are drawn "
    "(all links count in the numbers). Arsenal attack left to right in both matches.\n"
    "- **Average degree:** in + out connections per player, unweighted."
)

run = db.latest_run(ENTRY.slug)
results = db.results(run.id) if run else pd.DataFrame()
if run is None or results.empty:
    st.header("Results")
    empty_state("No published results yet. They appear here once the pipeline has run.")
    st.stop()


def series(metric: str, *dims: str) -> pd.DataFrame:
    """Rows of ``metric`` with the requested dims expanded into columns."""
    rows = results[results["metric"] == metric]
    out = pd.DataFrame({d: rows["dims"].map(lambda x, d=d: x.get(d)) for d in dims})
    out["value"] = rows["value"].to_numpy()
    return out.reset_index(drop=True)


matches = series("match_info", "match", "date", "home", "away", "home_score", "away_score")
if matches.empty:
    st.header("Results")
    empty_state("This run recorded no matches.")
    st.stop()
matches["match"] = matches["match"].astype(int)
matches["title"] = matches.apply(
    lambda m: f"{m.home} {m.home_score}-{m.away_score} {m.away}", axis=1
)
titles = dict(zip(matches["match"], matches["title"], strict=True))
summary = pd.concat(
    [
        series(m, "match").set_index("match")["value"].rename(m)
        for m in ("completed_passes", "players", "connections", "average_degree")
    ],
    axis=1,
)
players = series("passes_made", "match", "player", "label", "position", "x", "y", "received")
links = series("pass_link", "match", "passer", "recipient")


def stat(match_id: int, metric: str) -> float:
    """A per-match scalar, or NaN if this run did not record it."""
    try:
        return float(summary.at[match_id, metric])
    except KeyError:
        return float("nan")


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
cols = st.columns(len(matches))
for col, m in zip(cols, matches.itertuples(), strict=True):
    with col:
        st.markdown(f"**{m.title}** · {m.date}")
        with st.container(horizontal=True):
            st.metric("Completed passes", f"{stat(m.match, 'completed_passes'):,.0f}", border=True)
            st.metric("Players", f"{stat(m.match, 'players'):.0f}", border=True)
            st.metric("Connections", f"{stat(m.match, 'connections'):.0f}", border=True)
            st.metric("Avg degree", f"{stat(m.match, 'average_degree'):.2f}", border=True)

st.subheader("Passing network on the pitch")
with st.container(horizontal=True, vertical_alignment="bottom"):
    match_id = st.segmented_control(
        "Match",
        options=list(titles),
        format_func=titles.get,
        default=next(iter(titles)),
        key="match",
    )
    min_passes = st.slider("Draw links with at least", 1, 10, 3, key="min_passes")
match_id = match_id if match_id is not None else next(iter(titles))
st.plotly_chart(network_figure(match_id, min_passes), key="network", width="stretch")
st.caption(
    "Hover a player or a link for counts. Line width and opacity scale with passes on that "
    "link; arrows point from passer to recipient. Labels are common names or surnames."
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

st.header("Takeaways")
bullets = []
for m in matches.itertuples():
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
        f"({hub.value:.0f} of {stat(m.match, 'completed_passes'):.0f}). The busiest link was "
        f"{names[top.passer]} → {names[top.recipient]} ({top.value:.0f}); counting both "
        f"directions, that pair exchanged {pair:.0f} passes, "
        f"{pair / stat(m.match, 'completed_passes'):.0%} of the team's total."
    )
if len(matches) == 2:
    a, b = matches["match"].tolist()
    bullets.append(
        f"**Dense, not hub-and-spoke.** Each player linked to about "
        f"{stat(a, 'average_degree') / 2:.0f} team-mates in each direction in both matches "
        f"(average degree {stat(a, 'average_degree'):.2f} and {stat(b, 'average_degree'):.2f}). "
        "I read that as the ball moving through the whole side rather than through one "
        "playmaker, though two matches are far too few to call it a team style."
    )


def same_flank_pair(match_id: int) -> tuple[str, str] | None:
    """Labels of the busiest link if it joins a full-back and a midfielder on one side."""
    mp = players[players["match"] == match_id].set_index("player")
    ml = links[links["match"] == match_id]
    if ml.empty:
        return None
    top = ml.loc[ml["value"].idxmax()]
    pos = {mp.at[p, "position"] or "" for p in (top.passer, top.recipient) if p in mp.index}
    for side in ("Left", "Right"):
        if pos == {f"{side} Back", f"{side} Midfield"}:
            return mp.at[top.passer, "label"], mp.at[top.recipient, "label"]
    return None


flanks = {m.title: same_flank_pair(m.match) for m in matches.itertuples()}
if all(flanks.values()):
    pairs = "; ".join(f"{a} and {b} ({t})" for t, (a, b) in flanks.items())
    bullets.append(
        f"**The flanks carried the ball.** In every match here the busiest link joined a "
        f"full-back and the wide midfielder in front of him: {pairs}. I read that as Arsenal "
        "building play down the touchlines, but this analysis only counts passes. It says "
        "nothing about which passes created chances."
    )
st.markdown("\n".join(f"- {b}" for b in bullets))

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
theme = "dark" if st.context.theme.type == "dark" else "light"
st.image(ASSETS / f"statsbomb-logo-{theme}.png", width=220)
st.caption("Data: StatsBomb open data. Analysis and opinions are my own, not StatsBomb's.")
