import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import db, storage
from lab.registry import entry
from lab.ui import empty_state, page_header, provenance_footer

ENTRY = entry("enron-email-network")
CODE_PATHS = (
    "pipelines/ingest/snap.py",
    "pipelines/common/network_metrics.py",
    "pipelines/experiments/network_science/enron.py",
)
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated pair)

page_header(ENTRY)

st.header("Purpose")
st.write(
    "How did a large corporation organise its communication in the months before its 2001 "
    "collapse, and what did that structure cost it? Here I apply one framework (degree "
    "distribution, small-world properties, robustness to attack, community structure) to the "
    "Enron email network. It is Chapter 1 of my write-up *Conspiracies, Connectomes, and "
    "Cabernet*, a comparison of three real-world networks."
)

st.header("Method")
st.markdown(
    "- **Data:** SNAP's *email-Enron* graph. Nodes are email addresses (anonymised integers); "
    "an edge means at least one email was exchanged. **SNAP collapsed email direction into "
    "undirected edges**, so I know two addresses corresponded but not who wrote first.\n"
    "- **Degree distribution:** histogram of node degrees, then a least-squares line through "
    "log₁₀(count) vs log₁₀(degree), keeping only degrees seen more than once "
    "(singleton-filtered).\n"
    "- **Small world** (giant component): average clustering, and average shortest path from "
    "500 randomly sampled source nodes (seed 42; every BFS distance counts, including each "
    "source's zero to itself). Both are compared with an Erdős–Rényi graph of the same size: "
    "clustering *p = m / (n(n−1)/2)*, path *ln n / ln(np)*.\n"
    "- **Robustness** (giant component): remove one node at a time, either the current "
    "highest-degree node (targeted) or a uniformly random one (10 seeded trials), and track the "
    "largest component as a fraction of its original size. A separate test removes the 5 top "
    "hubs vs 5 random nodes (seed 42) from the full graph.\n"
    "- **Communities:** Louvain (python-louvain) on the giant component with seed 42. Louvain "
    "is randomised, so 20 further seeds are run to show how much the result varies.\n"
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


st.header("Results")
with st.container(horizontal=True):
    st.metric("Nodes", f"{value('nodes'):,.0f}", border=True)
    st.metric("Edges", f"{value('edges'):,.0f}", border=True)
    st.metric("Giant component", f"{value('giant_nodes'):,.0f}", border=True)
    st.metric("Max degree", f"{value('max_degree'):,.0f}", border=True)

st.subheader("Degree distribution")
deg = series("degree_count", "degree").astype({"degree": int})
with st.container(horizontal=True):
    st.metric("Power-law slope", f"{value('power_law_slope'):.2f}", border=True)
    st.metric("R²", f"{value('power_law_r2'):.3f}", border=True)
    st.metric("Average degree", f"{value('avg_degree'):.2f}", border=True)
left, right = st.columns(2)
with left:
    lo, hi = deg["degree"].min(), deg["degree"].max()
    hist = go.Figure(
        go.Histogram(
            x=deg["degree"],
            y=deg["value"],
            histfunc="sum",
            xbins={"start": lo, "end": hi, "size": (hi - lo) / 50},
            marker_color=BLUE,
            hovertemplate="Degree %{x}<br>%{y:,} nodes<extra></extra>",
        )
    )
    hist.update_layout(
        title="Histogram (50 bins)", xaxis_title="Degree", yaxis_title="Nodes", yaxis_type="log"
    )
    show(hist, "degree_hist")
with right:
    fit_pts = deg[deg["value"] > 1]
    x_fit = np.logspace(np.log10(fit_pts["degree"].min()), np.log10(fit_pts["degree"].max()), 100)
    y_fit = 10 ** value("power_law_intercept") * x_fit ** value("power_law_slope")
    loglog = go.Figure(
        [
            go.Scatter(
                x=fit_pts["degree"],
                y=fit_pts["value"],
                mode="markers",
                name="Observed",
                marker={"color": BLUE, "size": 8},
                hovertemplate="Degree %{x}<br>%{y:,} nodes<extra></extra>",
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
        xaxis={"type": "log", "title": "Degree"},
        yaxis={"type": "log", "title": "Nodes"},
        legend={"orientation": "h", "y": -0.2},
    )
    show(loglog, "degree_loglog")
st.caption(
    f"The fit uses the {value('power_law_points'):.0f} degrees seen more than once; "
    "degrees seen exactly once are omitted from the log-log chart."
)

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
        "Average clustering",
        f"{value('avg_clustering'):.3f}",
        f"random graph {value('random_clustering'):.6f}",
        delta_color="off",
        border=True,
    )
    st.metric("Clustering ratio", f"{value('clustering_ratio'):,.0f}×", border=True)

st.subheader("Robustness")
with st.container(horizontal=True):
    st.metric("Lost removing top 5 hubs", f"{value('hub_removal_loss'):,.0f} nodes", border=True)
    st.metric("Lost removing 5 random", f"{value('random_removal_loss'):,.0f} nodes", border=True)
    st.metric("Damage ratio", f"{value('damage_ratio'):,.1f}×", border=True)
curves = series("giant_fraction", "strategy", "seed", "step")
curves["removed"] = curves["step"] / value("giant_nodes")
targeted = curves[curves["strategy"] == "targeted"]
rand = curves[curves["strategy"] == "random"]
band = rand.groupby("removed")["value"].agg(["min", "max"]).reset_index()
first_seed = rand["seed"].min()
rand_main = rand[rand["seed"] == first_seed]
rob = go.Figure(
    [
        go.Scatter(
            x=band["removed"],
            y=band["max"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=band["removed"],
            y=band["min"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(42, 120, 214, 0.2)",
            name=f"Random: range of {rand['seed'].nunique()} trials",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=rand_main["removed"],
            y=rand_main["value"],
            mode="lines",
            name=f"Random failure (seed {first_seed:.0f})",
            line={"color": BLUE, "width": 2},
            hovertemplate="Removed %{x:.3%}<br>Giant %{y:.1%}<extra></extra>",
        ),
        go.Scatter(
            x=targeted["removed"],
            y=targeted["value"],
            mode="lines",
            name="Targeted attack (highest degree first)",
            line={"color": ORANGE, "width": 2},
            hovertemplate="Removed %{x:.3%}<br>Giant %{y:.1%}<extra></extra>",
        ),
    ]
)
rob.update_layout(
    xaxis={"title": "Fraction of giant component removed", "tickformat": ".2%"},
    yaxis={
        "title": "Giant component (fraction of original)",
        "range": [0, 1.05],
        "tickformat": ".0%",
    },
    legend={"orientation": "h", "y": -0.25},
)
show(rob, "robustness")
last = int(targeted["step"].max())
t_last = float(targeted.loc[targeted["step"] == last, "value"].iloc[0])
r_last = rand[rand["step"] == last]["value"]
st.caption(
    f"After {last} removals (the notebook's 100-step curve records the state before each "
    f"removal): targeted {1 - t_last:.1%} lost, "
    f"random {1 - r_last.max():.1%}–{1 - r_last.min():.1%} lost."
)

st.subheader("Communities")
with st.container(horizontal=True):
    st.metric("Communities", f"{value('communities'):,.0f}", border=True)
    st.metric("Modularity", f"{value('modularity'):.3f}", border=True)
    st.metric("Edges within communities", f"{value('within_fraction'):.1%}", border=True)
    st.metric("Median size", f"{value('median_community_size'):,.0f} nodes", border=True)
sizes = series("community_size", "rank")
size_fig = go.Figure(
    go.Bar(
        x=sizes["rank"],
        y=sizes["value"],
        marker_color=BLUE,
        hovertemplate="Rank %{x}<br>%{y:,} nodes<extra></extra>",
    )
)
size_fig.update_layout(
    title="Community sizes, largest to smallest",
    xaxis_title="Community rank",
    yaxis={"title": "Nodes", "type": "log"},
)
show(size_fig, "community_sizes")

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
        "within communities. My original, unseeded run (190 communities, modularity 0.608, 73.3%) "
        "falls inside this spread."
    )

st.markdown("**Top 30 communities**")
arts = {a.storage_path.rsplit("/", 1)[-1]: a for a in db.artifacts(run.id)}
nodes = (
    storage.read_parquet(arts["community_nodes.parquet"])
    if "community_nodes.parquet" in arts
    else None
)
edges = (
    storage.read_parquet(arts["community_edges.parquet"])
    if "community_edges.parquet" in arts
    else None
)
if nodes is None or edges is None or nodes.empty:
    empty_state("The community network is not available for this run.")
else:
    pos = nodes.set_index("community")[["x", "y"]]
    wmax = edges["weight"].max() if not edges.empty else 1
    traces: list[go.Scatter] = []
    # Edge width scales with cross-community edge count; one trace per width band.
    edges = edges.assign(band=np.ceil(5 * edges["weight"] / wmax).clip(1, 5))
    for b, grp in edges.groupby("band"):
        xs, ys = [], []
        for s, t in zip(grp["source"], grp["target"], strict=True):
            xs += [pos.at[s, "x"], pos.at[t, "x"], None]
            ys += [pos.at[s, "y"], pos.at[t, "y"], None]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": "rgba(128, 128, 128, 0.5)", "width": 0.6 * b},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    top10 = set(nodes.nlargest(10, "size")["community"])
    traces.append(
        go.Scatter(
            x=nodes["x"],
            y=nodes["y"],
            mode="markers+text",
            text=[f"C{c}" if c in top10 else "" for c in nodes["community"]],
            textposition="middle center",
            customdata=nodes[["community", "size"]],
            marker={
                "size": nodes["size"],
                "sizemode": "area",
                "sizeref": 2 * nodes["size"].max() / 60**2,
                "sizemin": 4,
                "color": nodes["size"],
                "colorscale": "Blues",
                "line": {"color": "white", "width": 2},
                "showscale": False,
            },
            hovertemplate="Community %{customdata[0]}<br>%{customdata[1]:,} nodes<extra></extra>",
            showlegend=False,
        )
    )
    net_fig = go.Figure(traces)
    net_fig.update_layout(
        height=560,
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
    )
    show(net_fig, "community_network")
    st.caption(
        "Node area = community size; edge width = number of edges between the two communities. "
        "Labels mark the 10 largest. Layout: spring layout (k = 15, seed 42), precomputed."
    )

pairs = series("cross_community_edges", "rank", "a", "a_size", "b", "b_size")
if not pairs.empty:
    with st.expander("Top cross-community connections"):
        st.dataframe(
            pairs.rename(
                columns={
                    "rank": "Rank",
                    "a": "Community A",
                    "a_size": "A nodes",
                    "b": "Community B",
                    "b_size": "B nodes",
                    "value": "Edges between",
                }
            ),
            hide_index=True,
            width="stretch",
        )

st.header("Takeaways")
mean_deg = value("avg_degree")
st.markdown(
    f"- **Scale-free, built around hubs.** Degrees follow a power law (slope "
    f"{value('power_law_slope'):.2f}, R² {value('power_law_r2'):.3f}). The busiest address has "
    f"degree {value('max_degree'):,.0f}, about {value('max_degree') / mean_deg:.0f}× the mean "
    f"of {mean_deg:.2f}. I read these hubs as executive assistants, shared inboxes or "
    "coordinators (the IDs are anonymised, so I can't confirm roles), and the pattern is what "
    "preferential attachment produces.\n"
    f"- **A small world.** Any two addresses in the giant component are about "
    f"{value('avg_path_sampled'):.1f} hops apart, yet clustering is "
    f"{value('clustering_ratio'):,.0f}× a random graph's: tight teams, bridged by hubs.\n"
    f"- **Robust to failure, fragile to attack.** Removing the 5 largest hubs cut "
    f"{value('hub_removal_loss'):,.0f} nodes from the giant component; 5 random nodes cut "
    f"{value('random_removal_loss'):,.0f}, a {value('damage_ratio'):,.0f}-fold difference. "
    f"Removing the top {last} hubs ({last / value('giant_nodes'):.2%} of the giant component) "
    f"shrinks it by {1 - t_last:.0%}; the same number of random removals barely registers.\n"
    f"- **The org chart, recovered.** Louvain finds {value('communities'):,.0f} communities "
    f"(modularity {value('modularity'):.3f}), and {value('within_fraction'):.0%} of edges stay "
    "inside them. My reading: this siloing put real structural barriers in the way of "
    "information, which may have helped contain knowledge of the fraud, even though a short "
    "path connected everyone in principle."
)

follow_up = entry("causal-emergence-worm-vs-email")
st.page_link(
    follow_up.path,
    label="Follow-up: do groups of addresses describe the network better than single addresses?",
    icon=follow_up.icon,
)

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
