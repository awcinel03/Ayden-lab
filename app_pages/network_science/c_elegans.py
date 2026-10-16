import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import db, storage
from lab.registry import entry
from lab.ui import empty_state, page_header, provenance_footer

ENTRY = entry("c-elegans-connectome")
CODE_PATHS = (
    "pipelines/ingest/snap.py",
    "pipelines/common/network_metrics.py",
    "pipelines/experiments/network_science/c_elegans.py",
)
# Categorical slots 1-4 (validated palette), assigned in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ROLE_COLORS = {"integrator": BLUE, "broadcaster": ORANGE, "relay": AQUA}
ROLE_LABELS = {
    "integrator": "Integrator (mostly receives)",
    "broadcaster": "Broadcaster (mostly sends)",
    "relay": "Relay (balanced)",
}

page_header(ENTRY)

st.header("Purpose")
st.write(
    "What did evolution optimise the worm's nervous system for? *C. elegans* was the first "
    "animal to have its neural wiring mapped. Here I apply the same framework as in the "
    "Enron entry (degree distribution, small-world properties, robustness, community "
    "structure) to the 131 neurons of its frontal ganglia, where every edge has a direction: "
    "the flow of a signal across a synapse. It is Chapter 2 of my write-up "
    "*Conspiracies, Connectomes, and Cabernet*."
)

st.header("Method")
st.markdown(
    "- **Data:** SNAP's *C. elegans frontal* network: a **directed** edge list (*u → v* = a "
    "synapse carrying signal from *u* to *v*) plus metadata giving each neuron's name and 2D "
    "anatomical position. Results use neuron names.\n"
    "- **Degree:** total degree = in + out. Histogram (20 bins) and a least-squares line "
    "through log₁₀(count) vs log₁₀(degree), keeping degrees seen more than once.\n"
    "- **Neuron roles:** share of a neuron's connections that are incoming, in / (in + out). "
    "Above 0.7 is an *integrator*, below 0.3 a *broadcaster*, otherwise a *relay*. Neurons "
    "with total degree above 20 are labelled as hubs.\n"
    "- **Small world:** directed clustering, and average shortest path from 50 randomly "
    "sampled source neurons (seed 42), following synapse direction. The random baseline is a "
    "directed Erdős–Rényi graph: *p = m / (n(n−1))*, path *ln n / ln(np)*.\n"
    "- **Robustness:** remove one neuron at a time and track the largest weakly connected "
    "component as a fraction of all 131 neurons. Strategies: highest total degree first "
    "(40 steps), highest in-degree and highest out-degree (30 steps each), and uniformly "
    "random (40 steps; seed 42 plus 9 more seeded trials for a range).\n"
    "- **Communities:** Louvain (python-louvain) on the **undirected projection** of the "
    "network, because Louvain does not handle direction; reciprocal synapses become one edge. "
    "Seed 42, plus 20 further seeds to show how much the result varies.\n"
    "- Computed offline with NetworkX; this page only reads stored results."
)

run = db.latest_run(ENTRY.slug)
results = db.results(run.id) if run else pd.DataFrame()
if run is None or results.empty:
    st.header("Results")
    empty_state("No published results yet. They appear here once the pipeline has run.")
    st.stop()

scalars = results[results["dims"].map(len) == 0].set_index("metric")["value"]


def value(metric: str) -> float:
    """A scalar result, or NaN if this run did not record it."""
    return float(scalars.get(metric, float("nan")))


def series(metric: str, *dims: str) -> pd.DataFrame:
    """Rows of ``metric`` with the requested dims expanded into columns."""
    rows = results[results["metric"] == metric]
    out = pd.DataFrame({d: rows["dims"].map(lambda x, d=d: x.get(d)) for d in dims})
    out["value"] = rows["value"].to_numpy()
    return out.reset_index(drop=True)


def show(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure with the lab's shared layout defaults."""
    fig.update_layout(margin={"l": 0, "r": 0, "t": 30, "b": 0}, hovermode="closest")
    st.plotly_chart(fig, key=key, width="stretch", theme="streamlit")


arts = {a.storage_path.rsplit("/", 1)[-1]: a for a in db.artifacts(run.id)}
neurons = storage.read_parquet(arts["neurons.parquet"]) if "neurons.parquet" in arts else None
synapses = storage.read_parquet(arts["synapses.parquet"]) if "synapses.parquet" in arts else None
have_graph = neurons is not None and synapses is not None and not neurons.empty

st.header("Results")
with st.container(horizontal=True):
    st.metric("Neurons", f"{value('nodes'):,.0f}", border=True)
    st.metric("Synapses (directed)", f"{value('edges'):,.0f}", border=True)
    st.metric("Mean degree", f"{value('mean_degree'):.2f}", border=True)
    st.metric("Max degree", f"{value('max_degree'):,.0f}", border=True)

st.subheader("Anatomical network")
if not have_graph:
    empty_state("The neuron map is not available for this run.")
else:
    placed = neurons.dropna(subset=["x", "y"]).set_index("neuron")
    edges = synapses[synapses["source"].isin(placed.index) & synapses["target"].isin(placed.index)]
    xs, ys, ax, ay, angles = [], [], [], [], []
    for s, t in zip(edges["source"], edges["target"], strict=True):
        x0, y0, x1, y1 = placed.at[s, "x"], placed.at[s, "y"], placed.at[t, "x"], placed.at[t, "y"]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        ax.append(x0 + 0.8 * (x1 - x0))
        ay.append(y0 + 0.8 * (y1 - y0))
        angles.append(math.degrees(math.atan2(x1 - x0, y1 - y0)))  # clockwise from up
    traces: list[go.Scatter] = [
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "rgba(128, 128, 128, 0.25)", "width": 0.6},
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=ax,
            y=ay,
            mode="markers",
            marker={
                "symbol": "triangle-up",
                "size": 5,
                "angle": angles,
                "color": "rgba(128, 128, 128, 0.5)",
            },
            hoverinfo="skip",
            showlegend=False,
        ),
    ]
    for role, color in ROLE_COLORS.items():
        grp = placed[placed["role"] == role]
        traces.append(
            go.Scatter(
                x=grp["x"],
                y=grp["y"],
                mode="markers+text",
                name=ROLE_LABELS[role],
                text=[n if h else "" for n, h in zip(grp.index, grp["hub"], strict=True)],
                textposition="top center",
                textfont={"size": 10},
                customdata=np.stack(
                    [grp.index, grp["in_degree"], grp["out_degree"], grp["total_degree"]], axis=-1
                ),
                marker={
                    "color": color,
                    "size": 6 + 0.6 * grp["total_degree"],
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>in %{customdata[1]} · out %{customdata[2]}"
                    f"<br>{role}<extra></extra>"
                ),
            )
        )
    net = go.Figure(traces)
    net.update_layout(
        height=560,
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        legend={"orientation": "h", "y": -0.05},
    )
    show(net, "anatomical_network")
    toward, away = value("signal_toward_centre"), value("signal_away_from_centre")
    st.caption(
        "Neurons at their anatomical positions. Size = total degree; colour = role; labels mark "
        "hubs (total degree above 20); arrowheads show signal direction. "
        f"{toward / (toward + away):.1%} of synapses carry signal toward the anatomical centre "
        f"and {away / (toward + away):.1%} away from it."
    )

st.subheader("Neuron roles")
with st.container(horizontal=True):
    roles = series("role_count", "role").set_index("role")["value"]
    for role in ROLE_COLORS:
        st.metric(ROLE_LABELS[role].split(" (")[0] + "s", f"{roles.get(role, 0):,.0f}", border=True)
if have_graph:
    hi = int(max(neurons["in_degree"].max(), neurons["out_degree"].max())) + 1
    scatter = go.Figure(
        [
            go.Scatter(
                x=[0, hi * 3 / 7],
                y=[0, hi],
                mode="lines",
                line={"color": "rgba(128, 128, 128, 0.6)", "dash": "dot", "width": 1},
                name="in / (in + out) = 0.7",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[0, hi],
                y=[0, hi * 3 / 7],
                mode="lines",
                line={"color": "rgba(128, 128, 128, 0.6)", "dash": "dash", "width": 1},
                name="in / (in + out) = 0.3",
                hoverinfo="skip",
            ),
        ]
        + [
            go.Scatter(
                x=grp["out_degree"],
                y=grp["in_degree"],
                mode="markers",
                name=ROLE_LABELS[role],
                text=grp["neuron"],
                marker={
                    "color": color,
                    "size": 9,
                    "opacity": 0.85,
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate="<b>%{text}</b><br>out %{x} · in %{y}<extra></extra>",
            )
            for role, color in ROLE_COLORS.items()
            for grp in [neurons[neurons["role"] == role]]
        ]
    )
    scatter.update_layout(
        xaxis={"title": "Out-degree (signals sent)", "range": [-0.5, hi]},
        yaxis={"title": "In-degree (signals received)", "range": [-0.5, hi]},
        legend={"orientation": "h", "y": -0.2},
        height=480,
    )
    show(scatter, "in_out_scatter")
    st.caption("Many neurons share a degree pair; hover a point for the neurons it represents.")

st.subheader("Degree distribution")
deg = series("degree_count", "degree").astype({"degree": int})
with st.container(horizontal=True):
    st.metric("Power-law slope", f"{value('power_law_slope'):.2f}", border=True)
    st.metric("R²", f"{value('power_law_r2'):.3f}", border=True)
    st.metric("Median degree", f"{value('median_degree'):.0f}", border=True)
left, right = st.columns(2)
with left:
    lo, hi_deg = deg["degree"].min(), deg["degree"].max()
    hist = go.Figure(
        go.Histogram(
            x=deg["degree"],
            y=deg["value"],
            histfunc="sum",
            xbins={"start": lo, "end": hi_deg, "size": (hi_deg - lo) / 20},
            marker_color=BLUE,
            hovertemplate="Degree %{x}<br>%{y} neurons<extra></extra>",
        )
    )
    hist.update_layout(
        title="Histogram (20 bins)", xaxis_title="Total degree", yaxis_title="Neurons"
    )
    show(hist, "degree_hist")
with right:
    pts = deg[deg["value"] > 1]
    x_fit = np.logspace(np.log10(pts["degree"].min()), np.log10(pts["degree"].max()), 100)
    y_fit = 10 ** value("power_law_intercept") * x_fit ** value("power_law_slope")
    loglog = go.Figure(
        [
            go.Scatter(
                x=pts["degree"],
                y=pts["value"],
                mode="markers",
                name="Observed",
                marker={"color": BLUE, "size": 8},
                hovertemplate="Degree %{x}<br>%{y} neurons<extra></extra>",
            ),
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=f"Fit (slope {value('power_law_slope'):.2f})",
                line={"color": ORANGE, "width": 2, "dash": "dash"},
                hoverinfo="skip",
            ),
        ]
    )
    loglog.update_layout(
        title="Log-log with power-law fit",
        xaxis={"type": "log", "title": "Total degree"},
        yaxis={"type": "log", "title": "Neurons"},
        legend={"orientation": "h", "y": -0.2},
    )
    show(loglog, "degree_loglog")

st.subheader("Small world")
with st.container(horizontal=True):
    st.metric(
        "Average path (hops)",
        f"{value('avg_path_sampled'):.2f}",
        f"random graph {value('random_path'):.2f}",
        delta_color="off",
        border=True,
    )
    st.metric(
        "Average clustering (directed)",
        f"{value('avg_clustering'):.4f}",
        f"random graph {value('random_clustering'):.4f}",
        delta_color="off",
        border=True,
    )
    st.metric("Clustering ratio", f"{value('clustering_ratio'):.1f}×", border=True)

st.subheader("Robustness")
curves = series("giant_fraction", "strategy", "seed", "step")
rand = curves[curves["strategy"] == "random"]
band = rand.groupby("step")["value"].agg(["min", "max"]).reset_index()
first_seed = rand["seed"].min()
lines = [  # random first so the targeted curves draw on top of it
    ("random", first_seed, f"Random failure (seed {first_seed:.0f})", BLUE),
    ("targeted", None, "Targeted: highest total degree", ORANGE),
    ("in_degree", None, "Targeted: highest in-degree (integrators)", AQUA),
    ("out_degree", None, "Targeted: highest out-degree (broadcasters)", YELLOW),
]
rob = go.Figure(
    [
        go.Scatter(
            x=band["step"],
            y=band["max"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=band["step"],
            y=band["min"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(42, 120, 214, 0.2)",
            name=f"Random: range of {rand['seed'].nunique()} trials",
            hoverinfo="skip",
        ),
    ]
    + [
        go.Scatter(
            x=c["step"],
            y=c["value"],
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 2},
            marker={"size": 4},
            hovertemplate="%{x} removed<br>Giant %{y:.1%}<extra>" + label + "</extra>",
        )
        for strategy, seed, label, color in lines
        for c in [
            curves[
                (curves["strategy"] == strategy)
                & (curves["seed"].isna() if seed is None else curves["seed"] == seed)
            ]
        ]
    ]
)
rob.update_layout(
    xaxis={"title": "Neurons removed"},
    yaxis={
        "title": "Largest component (fraction of 131 neurons)",
        "range": [0, 1.05],
        "tickformat": ".0%",
    },
    legend={"orientation": "h", "y": -0.25},
    height=480,
)
show(rob, "robustness")


def at(strategy: str, step: int, seed: float | None = None) -> float:
    """Giant fraction of one curve at ``step``, or NaN if the curve is shorter."""
    c = curves[(curves["strategy"] == strategy) & (curves["step"] == step)]
    c = c[c["seed"].isna()] if seed is None else c[c["seed"] == seed]
    return float(c["value"].iloc[0]) if len(c) else float("nan")


t30, r30 = at("targeted", 30), at("random", 30, first_seed)
in_out_gap = max(
    abs(at("in_degree", s) - at("out_degree", s))
    for s in range(int(curves[curves["strategy"] == "in_degree"]["step"].max()) + 1)
)
st.caption(
    f"After 30 removals: targeted {t30:.1%}, random {r30:.1%} of neurons remain connected. "
    f"In-degree and out-degree targeting never differ by more than {in_out_gap:.1%}."
)

st.subheader("Communities")
with st.container(horizontal=True):
    st.metric("Communities", f"{value('communities'):,.0f}", border=True)
    st.metric("Modularity", f"{value('modularity'):.3f}", border=True)
    st.metric("Edges within communities", f"{value('within_fraction'):.1%}", border=True)
sizes = series("community_size", "rank")
left, right = st.columns(2)
with left:
    size_fig = go.Figure(
        go.Bar(
            x=sizes["rank"],
            y=sizes["value"],
            marker_color=BLUE,
            hovertemplate="Rank %{x}<br>%{y} neurons<extra></extra>",
        )
    )
    size_fig.update_layout(
        title="Community sizes", xaxis_title="Community rank", yaxis_title="Neurons"
    )
    show(size_fig, "community_sizes")
with right:
    hubs = series("hub_community", "rank", "neuron", "community", "community_size")
    st.markdown("**Hub neurons and their communities**")
    st.dataframe(
        hubs.rename(
            columns={
                "rank": "Rank",
                "neuron": "Neuron",
                "value": "Total degree",
                "community": "Community",
                "community_size": "Community size",
            }
        )[["Rank", "Neuron", "Total degree", "Community", "Community size"]],
        hide_index=True,
        width="stretch",
    )
sweep = {
    m: series(f"sweep_{m}", "seed")["value"]
    for m in ("communities", "modularity", "within_fraction")
}
if all(len(v) for v in sweep.values()):
    st.caption(
        f"Louvain is randomised. Across {len(sweep['communities'])} other seeds: "
        f"{sweep['communities'].min():.0f}–{sweep['communities'].max():.0f} communities, "
        f"modularity {sweep['modularity'].min():.3f}–{sweep['modularity'].max():.3f}, "
        f"{sweep['within_fraction'].min():.1%}–{sweep['within_fraction'].max():.1%} of edges "
        "within communities. Community numbers are arbitrary labels."
    )

st.header("Takeaways")
bullets = [
    f"**No power law, no extreme hubs.** The log-log fit is flat and meaningless (slope "
    f"{value('power_law_slope'):.2f}, R² {value('power_law_r2'):.3f}). Degrees cluster around "
    f"the mean of {value('mean_degree'):.2f} (median {value('median_degree'):.0f}), and the "
    f"busiest neuron has degree {value('max_degree'):.0f}, only "
    f"{value('max_degree') / value('mean_degree'):.1f}× the mean. I read this as evolution "
    "favouring a distributed nervous system over a hub-dependent one.",
]
if have_graph:
    by = neurons.set_index("neuron")
    top_in, top_out = by["in_degree"].idxmax(), by["out_degree"].idxmax()
    broadcasters = [n for n in ("AVAL", "AVEL") if n in by.index]
    pair = " and ".join(
        f"{n} (in {by.at[n, 'in_degree']}, out {by.at[n, 'out_degree']})" for n in broadcasters
    )
    bullets.append(
        f"**Direction reveals roles.** {top_in} receives the most synapses (in-degree "
        f"{by.at[top_in, 'in_degree']}); {top_out} sends the most (out-degree "
        f"{by.at[top_out, 'out_degree']}) and also receives {by.at[top_out, 'in_degree']}, the "
        "profile of an interneuron that gathers and redistributes signals. "
        + (f"{pair} send far more than they receive. " if pair else "")
        + "I connect these profiles to known biology (relay interneurons; command "
        "interneurons that drive locomotion), but that link comes from the neuroscience "
        "literature, not from anything this analysis tests."
    )
bullets += [
    f"**Small world without hubs.** Neurons are {value('avg_path_sampled'):.2f} synaptic hops "
    f"apart on average (a random directed graph: {value('random_path'):.2f}), and clustering is "
    f"{value('clustering_ratio'):.1f}× random, far less than a human organisation's. SNAP reports "
    "a 90th-percentile effective diameter of 3.",
    f"**Damage-tolerant.** Removing the 30 best-connected neurons leaves {t30:.0%} connected; "
    f"30 random removals leave {r30:.0%}. Targeting integrators or broadcasters makes almost no "
    "difference. There is no single point of failure.",
    f"**Integrated circuits.** Louvain finds {value('communities'):.0f} communities of fairly even "
    f"size (modularity {value('modularity'):.3f}), and {1 - value('within_fraction'):.0%} of "
    "connections cross community boundaries. I suspect the communities correspond to "
    "functional circuits (locomotion, chemosensation, feeding), though confirming that would "
    "need the neuroscience literature.",
]
st.markdown("\n".join(f"- {b}" for b in bullets))

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
