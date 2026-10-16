import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import db, storage
from lab.registry import entry
from lab.ui import empty_state, page_header, provenance_footer

ENTRY = entry("wine-flavor-network")
CODE_PATHS = (
    "pipelines/ingest/wine_flavor.py",
    "pipelines/ingest/wine_flavor_lexicon_v1.json",
    "pipelines/common/network_metrics.py",
    "pipelines/experiments/network_science/wine_flavor.py",
)
# Categorical slots 1-3 (validated palette), assigned to communities by size rank.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COMMUNITY_COLORS = {1: BLUE, 2: ORANGE, 3: AQUA}
OTHER_COLOR = "#8c8c8c"
# Marker terms for the red/white reading in Takeaways (checked against the partition).
RED_MARKERS = ("cherry", "tannin", "blackberry", "plum")
WHITE_MARKERS = ("apple", "citrus", "lemon", "pear")

page_header(ENTRY)

st.header("Purpose")
st.write(
    "Which flavours do professional tasters perceive and describe together? Unlike my other "
    "network entries, this network isn't downloaded as an edge list. I built it from the "
    "language of 129,971 Wine Enthusiast tasting notes, so it captures the structure of expert "
    "sensory vocabulary rather than physical connections or messages. I then apply the same "
    "framework as in the Enron and *C. elegans* entries (degree distribution, small-world "
    "properties, robustness, community structure). It is Chapter 3 of my write-up "
    "*Conspiracies, Connectomes, and Cabernet*."
)

st.header("Method")
st.markdown(
    "- **Data:** the Kaggle *Wine Reviews* dataset (winemag-data-130k-v2, reviews scraped from "
    "Wine Enthusiast). The review text belongs to Wine Enthusiast, so it never leaves my "
    "machine: the pipeline publishes only derived counts (how many reviews mention each term, "
    "and each pair of terms). No review text is stored or shown here.\n"
    "- **Lexicon:** 116 descriptors I took from the UC Davis Wine Aroma Wheel (fruit, floral, "
    "spice, earth, oak, plus structural words such as *acid*, *acidity*, *tannin* and "
    "*fresh*), kept exactly as in my notebook.\n"
    "- **Matching:** a term counts if it appears anywhere in the lower-cased review "
    "(substring, no word boundaries), as in my notebook. So *acid* also matches inside "
    "*acidity* and *tar* inside *tart*. Reviews with fewer than two terms are skipped.\n"
    "- **Network:** one node per term found in at least one review; an edge joins two terms "
    "if at least **10** reviews mention both, weighted by that number of reviews. A term "
    "whose pairs all fall below 10 reviews stays in as an isolated node.\n"
    "- **Weights:** only the Louvain communities (and their modularity) use edge weights. "
    "Degree, clustering, shortest paths and robustness treat every edge alike.\n"
    "- **Degree:** histogram (20 bins) and a least-squares line through log₁₀(count) vs "
    "log₁₀(degree), keeping degrees seen more than once (the same rule as my Enron and "
    "*C. elegans* entries, so the fits compare directly). My notebook's own fit was undefined: "
    "it kept every degree with a nonzero count, including degree 0 (the isolated *brett*), "
    "and log₁₀(0) made the slope and R² NaN. The slope 0.06 and R² 0.010 in my write-up were "
    "copied from the *C. elegans* chapter.\n"
    "- **Small world:** unweighted clustering, and average shortest path from 50 randomly "
    "sampled source terms (seed 42; each source's zero distance to itself is included, as in "
    "my notebook). Baseline: an Erdős–Rényi graph with the same density, "
    "*p = m / (n(n−1)/2)*, path *ln n / ln(np)*.\n"
    "- **Robustness:** remove one term at a time for 50 steps and track the largest connected "
    "component as a fraction of all terms, recording the state before each removal. Targeted "
    "removes the current highest-degree term (first in node order on ties); random draws "
    "uniformly with seed 42, as in my notebook, plus 9 more seeded trials for a range.\n"
    "- **Communities:** weighted Louvain (python-louvain), seed 42, plus 20 further seeds to "
    "show how much the result varies. Communities are numbered by size.\n"
    "- **Shortest paths:** precomputed for every pair of terms in **unweighted hops** (one "
    "shortest path per pair; there can be ties).\n"
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


def table(name: str) -> pd.DataFrame | None:
    """One of the run's Parquet artifacts, or ``None`` if it is unavailable."""
    return storage.read_parquet(arts[f"{name}.parquet"]) if f"{name}.parquet" in arts else None


flavors = table("flavors")
cooccurrence = table("cooccurrence")
paths = table("shortest_paths")
have_graph = flavors is not None and cooccurrence is not None and not flavors.empty

terms = series("term_reviews", "term", "degree").astype({"degree": int})
n_terms = value("nodes")

st.header("Results")
with st.container(horizontal=True):
    st.metric("Flavor terms", f"{n_terms:,.0f}", border=True)
    st.metric("Co-occurrence edges", f"{value('edges'):,.0f}", border=True)
    st.metric(
        "Average degree", f"{value('avg_degree'):.2f}", f"of {n_terms - 1:.0f} possible",
        delta_color="off", border=True,
    )  # fmt: skip
    st.metric("Density", f"{value('density'):.3f}", border=True)

st.subheader("Most frequent descriptors")
top_n = st.slider("Show the top", 5, min(40, max(5, len(terms))), min(15, len(terms)), key="top_n")
freq = terms.nlargest(top_n, "value").sort_values("value")
freq_fig = go.Figure(
    go.Bar(
        x=freq["value"],
        y=freq["term"],
        orientation="h",
        marker_color=BLUE,
        customdata=freq["degree"],
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} reviews · degree %{customdata}<extra></extra>",
    )
)
freq_fig.update_layout(height=max(300, 22 * top_n), xaxis_title="Reviews mentioning the term")
show(freq_fig, "frequencies")
pairs = series("top_pair", "rank", "term_a", "term_b")
if not pairs.empty:
    with st.expander(f"Strongest {len(pairs)} pairings"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Rank": pairs["rank"].astype(int),
                    "Pair": pairs["term_a"] + " + " + pairs["term_b"],
                    "Reviews with both": pairs["value"].astype(int),
                }
            ),
            hide_index=True,
            width="stretch",
        )

st.subheader("Degree distribution")
deg = series("degree_count", "degree").astype({"degree": int})
with st.container(horizontal=True):
    st.metric("Power-law slope", f"{value('power_law_slope'):.2f}", border=True)
    st.metric("R²", f"{value('power_law_r2'):.3f}", border=True)
    st.metric("Max degree", f"{value('max_degree'):.0f}", border=True)
    st.metric("Isolated terms", f"{value('isolated_nodes'):.0f}", border=True)
left, right = st.columns(2)
with left:
    lo, hi_deg = deg["degree"].min(), deg["degree"].max()
    hist = go.Figure(
        go.Histogram(
            x=deg["degree"],
            y=deg["value"],
            histfunc="sum",
            xbins={"start": lo, "end": hi_deg, "size": max(1, (hi_deg - lo) / 20)},
            marker_color=BLUE,
            hovertemplate="Degree %{x}<br>%{y} terms<extra></extra>",
        )
    )
    hist.update_layout(title="Histogram (20 bins)", xaxis_title="Degree", yaxis_title="Terms")
    show(hist, "degree_hist")
with right:
    pts = deg[(deg["value"] > 1) & (deg["degree"] > 0)]
    traces = [
        go.Scatter(
            x=pts["degree"],
            y=pts["value"],
            mode="markers",
            name="Observed (count > 1)",
            marker={"color": BLUE, "size": 8},
            hovertemplate="Degree %{x}<br>%{y} terms<extra></extra>",
        )
    ]
    if len(pts) > 1 and np.isfinite(value("power_law_slope")):
        x_fit = np.logspace(np.log10(pts["degree"].min()), np.log10(pts["degree"].max()), 100)
        traces.append(
            go.Scatter(
                x=x_fit,
                y=10 ** value("power_law_intercept") * x_fit ** value("power_law_slope"),
                mode="lines",
                name=f"Fit (slope {value('power_law_slope'):.2f})",
                line={"color": ORANGE, "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    loglog = go.Figure(traces)
    loglog.update_layout(
        title="Log-log with power-law fit",
        xaxis={"type": "log", "title": "Degree"},
        yaxis={"type": "log", "title": "Terms"},
        legend={"orientation": "h", "y": -0.2},
    )
    show(loglog, "degree_loglog")

st.subheader("Small world")
with st.container(horizontal=True):
    st.metric(
        "Average path (hops)", f"{value('avg_path_sampled'):.2f}",
        f"random graph {value('random_path'):.2f}", delta_color="off", border=True,
    )  # fmt: skip
    st.metric(
        "Average clustering", f"{value('avg_clustering'):.3f}",
        f"random graph {value('random_clustering'):.3f}", delta_color="off", border=True,
    )  # fmt: skip
    st.metric("Clustering ratio", f"{value('clustering_ratio'):.1f}×", border=True)

st.subheader("Flavor network")
sizes = series("community_size", "rank")
if not have_graph:
    empty_state("The flavor network is not available for this run.")
else:
    threshold = value("view_edge_threshold")
    pos = flavors.set_index("term")
    strong = cooccurrence[cooccurrence["weight"] > threshold]
    xs, ys = [], []
    for a, b in zip(strong["term_a"], strong["term_b"], strict=True):
        xs += [pos.at[a, "x"], pos.at[b, "x"], None]
        ys += [pos.at[a, "y"], pos.at[b, "y"], None]
    traces = [
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "rgba(128, 128, 128, 0.3)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    ]
    for comm in sorted(flavors["community"].unique()):
        grp = flavors[flavors["community"] == comm]
        leaders = ", ".join(grp.nlargest(3, "reviews")["term"])
        traces.append(
            go.Scatter(
                x=grp["x"],
                y=grp["y"],
                mode="markers+text",
                name=f"Community {comm} ({len(grp)} terms: {leaders}…)"
                if len(grp) > 3
                else f"Community {comm} ({', '.join(grp['term'])})",
                text=grp["term"],
                textposition="top center",
                textfont={"size": 9},
                customdata=np.stack([grp["reviews"], grp["degree"]], axis=-1),
                marker={
                    "color": COMMUNITY_COLORS.get(int(comm), OTHER_COLOR),
                    "size": 6 + 10 * grp["degree"] / max(1, n_terms - 1),
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>%{customdata[0]:,} reviews · degree %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
    net = go.Figure(traces)
    net.update_layout(
        height=640,
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        legend={"orientation": "h", "y": -0.05},
    )
    show(net, "flavor_network")
    st.caption(
        f"Every term, coloured by Louvain community (seed 42), sized by degree. To keep it "
        f"readable, only the {len(strong):,} pairs mentioned together in more than "
        f"{threshold:,.0f} reviews are drawn, and the layout (spring layout, seed 42) uses only "
        "those pairs, as in my notebook, so a term's position says little about its weaker "
        "links. Hover a term for its counts."
    )

st.subheader("Robustness")
curves = series("giant_fraction", "strategy", "seed", "step")
rand = curves[curves["strategy"] == "random"]
band = rand.groupby("step")["value"].agg(["min", "max"]).reset_index()
first_seed = rand["seed"].min()
rob = go.Figure(
    [
        go.Scatter(
            x=band["step"],
            y=band["max"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        ),  # fmt: skip
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
        for strategy, seed, label, color in (
            ("random", first_seed, f"Random failure (seed {first_seed:.0f})", BLUE),
            ("targeted", None, "Targeted: highest degree first", ORANGE),
        )
        for c in [
            curves[
                (curves["strategy"] == strategy)
                & (curves["seed"].isna() if seed is None else curves["seed"] == seed)
            ]
        ]
    ]
)
rob.update_layout(
    xaxis={"title": "Flavor terms removed"},
    yaxis={
        "title": f"Largest component (fraction of {n_terms:.0f} terms)",
        "range": [0, 1.05],
        "tickformat": ".0%",
    },
    legend={"orientation": "h", "y": -0.25},
    height=440,
)
show(rob, "robustness")


def at(strategy: str, step: int, seed: float | None = None) -> float:
    """Giant fraction of one curve at ``step``, or NaN if the curve is shorter."""
    c = curves[(curves["strategy"] == strategy) & (curves["step"] == step)]
    c = c[c["seed"].isna()] if seed is None else c[c["seed"] == seed]
    return float(c["value"].iloc[0]) if len(c) else float("nan")


t10, r10, t25, r25 = (
    at("targeted", 10), at("random", 10, first_seed),
    at("targeted", 25), at("random", 25, first_seed),
)  # fmt: skip


def lost_beyond(frac: float, k: int) -> float:
    """Terms outside the largest component after ``k`` removals, not counting those removed."""
    return (1 - frac) * n_terms - k


st.caption(
    f"After 10 removals: targeted {t10:.3f}, random {r10:.3f}. After 25: targeted {t25:.3f}, "
    f"random {r25:.3f}. The fractions are of all {n_terms:.0f} terms, so they include the "
    "removed terms and any term that was isolated from the start."
)

st.subheader("Communities")
with st.container(horizontal=True):
    st.metric("Communities", f"{value('communities'):,.0f}", border=True)
    st.metric("Modularity (weighted)", f"{value('modularity'):.3f}", border=True)
    st.metric("Edges within communities", f"{value('within_fraction'):.1%}", border=True)
if have_graph:
    members = (
        flavors.sort_values("reviews", ascending=False)
        .groupby("community", sort=True)["term"]
        .agg(lambda s: ", ".join(s))
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Community": members.index.astype(int),
                "Terms": flavors.groupby("community").size().reindex(members.index).to_numpy(),
                "Members (most frequent first)": members.to_numpy(),
            }
        ),
        hide_index=True,
        width="stretch",
    )
sweep = {
    m: series(f"sweep_{m}", "seed")["value"]
    for m in ("communities", "modularity", "largest_community", "second_community")
}
if all(len(v) for v in sweep.values()):

    def span(v: pd.Series, fmt: str = ".0f") -> str:
        """``min–max`` of a sweep, or the single value when every seed agrees."""
        lo, hi = format(v.min(), fmt), format(v.max(), fmt)
        return lo if lo == hi else f"{lo}–{hi}"

    st.caption(
        f"Louvain is randomised. Across {len(sweep['communities'])} other seeds: "
        f"{span(sweep['communities'])} communities, modularity "
        f"{span(sweep['modularity'], '.3f')}, largest community "
        f"{span(sweep['largest_community'])} terms, second {span(sweep['second_community'])}."
    )

st.subheader("Shortest path between two flavors")
if paths is None or not have_graph:
    empty_state("Shortest paths are not available for this run.")
else:
    names = sorted(flavors["term"])
    default_b = names.index("lemon") if "lemon" in names else min(1, len(names) - 1)
    with st.container(horizontal=True):
        a = st.selectbox("From", names, index=names.index("cherry") if "cherry" in names else 0)
        b = st.selectbox("To", names, index=default_b)
    if a == b:
        st.write("Pick two different flavors.")
    else:
        s, t = sorted((a, b))
        hit = paths[(paths["source"] == s) & (paths["target"] == t)]
        if hit.empty:
            lonely = [x for x in (a, b) if flavors.set_index("term").at[x, "degree"] == 0]
            st.write(
                f"No path: **{' and '.join(lonely) or a}** has no co-occurrence edge "
                "(no pair reaches 10 reviews), so it is not connected to anything."
            )
        else:
            steps = hit["path"].iloc[0].split(" > ")
            steps = steps if steps[0] == a else steps[::-1]
            hops = int(hit["hops"].iloc[0])
            st.markdown(
                " → ".join(f"**{x}**" for x in steps) + f"  ({hops} hop{'s' if hops != 1 else ''})"
            )
            if hops == 1:
                w = cooccurrence[
                    ((cooccurrence["term_a"] == a) & (cooccurrence["term_b"] == b))
                    | ((cooccurrence["term_a"] == b) & (cooccurrence["term_b"] == a))
                ]["weight"]
                if len(w):
                    st.caption(f"Mentioned together in {int(w.iloc[0]):,} reviews.")
            else:
                st.caption(
                    "Fewer than 10 reviews mention both, so they connect through a shared "
                    "neighbour. Another path of the same length may exist."
                )
    hop_counts = series("path_hops_count", "hops")
    if not hop_counts.empty:
        total = hop_counts["value"].sum()
        direct = hop_counts.loc[hop_counts["hops"] == 1, "value"].sum()
        st.caption(
            f"Of {total:,.0f} connected pairs, {direct / total:.1%} are one hop apart and the "
            f"longest shortest path is {hop_counts['hops'].max():.0f} hops."
        )

st.header("Takeaways")
iso = terms[terms["degree"] == 0]
linked = terms[terms["degree"] > 0]
tail = ", ".join(linked.nsmallest(4, "degree")["term"])
top_freq = terms.nlargest(4, "value")
top_deg = terms.sort_values("degree", ascending=False, kind="stable").head(4)
bullets = [
    f"**Near-complete, not scale-free.** The average term co-occurs with "
    f"{value('avg_degree'):.2f} of the {n_terms - 1:.0f} others (density "
    f"{value('density'):.3f}). The log-log fit is flat and meaningless (slope "
    f"{value('power_law_slope'):.2f}, R² {value('power_law_r2'):.3f}). The most connected term "
    f"has degree {value('max_degree'):.0f}; the least connected terms ({tail}) form a "
    "long tail on the low side of the histogram.",
    "**Structure dominates.** The most frequent terms are "
    + ", ".join(f"{r.term} ({r.value:,.0f} reviews)" for r in top_freq.itertuples())
    + "; the best connected are "
    + ", ".join(f"{r.term} ({r.degree})" for r in top_deg.itertuples())
    + ". I read this as structure and flavour being inseparable in how tasters write: "
    "acidity shapes how fruit is perceived, so the two are described together.",
    f"**Small world, trivially.** Terms are {value('avg_path_sampled'):.2f} hops apart on "
    f"average (random graph {value('random_path'):.2f}) and clustering is only "
    f"{value('clustering_ratio'):.1f}× random. At this density a random graph looks almost "
    "the same, so the small-world label says little here.",
    f"**No critical hubs.** After 10 removals the largest component holds {t10:.1%} of terms "
    f"under targeted attack and {r10:.1%} under random failure; after 25, {t25:.1%} and "
    f"{r25:.1%}. Beyond the 25 terms removed, only {lost_beyond(t25, 25):.0f} (targeted) and "
    f"{lost_beyond(r25, 25):.0f} (random) fall outside the largest component, counting any "
    "term isolated from the start. The network degrades by simple subtraction.",
]
if have_graph:
    part = flavors.set_index("term")["community"]
    red = {part[t] for t in RED_MARKERS if t in part.index}
    white = {part[t] for t in WHITE_MARKERS if t in part.index}
    split = len(red) == 1 and len(white) == 1 and red != white
    sentence = (
        f"**Communities.** Louvain finds {value('communities'):.0f} communities with modularity "
        f"{value('modularity'):.3f}, sizes " + ", ".join(f"{v:.0f}" for v in sizes["value"]) + "."
    )
    if split:
        r, w = red.pop(), white.pop()
        sentence += (
            f" {', '.join(RED_MARKERS)} all fall in community {r}, and "
            f"{', '.join(WHITE_MARKERS)} in community {w}. I read this as the red/white divide, "
            "recovered from co-occurrence alone; the algorithm knows nothing about wine."
        )
    bullets.append(sentence)
if len(iso):
    names_iso = ", ".join(f"{r.term} ({r.value:,.0f} reviews)" for r in iso.itertuples())
    bullets.append(
        f"**Isolated:** {names_iso}. No pair involving it reaches the 10-review threshold, so it "
        "has no edges and Louvain gives it a community of its own. Where it lands in the "
        "picture says nothing about red or white."
    )
bullets.append(
    "**Flavour as a whole.** My interpretation: a network this dense suggests experts don't "
    "describe flavours as independent attributes but as an integrated impression (oak with "
    "vanilla because barrel ageing gives both, mineral with citrus because both mark the same "
    "styles). The network shows the co-occurrence, not the reason for it."
)
st.markdown("\n".join(f"- {b}" for b in bullets))

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
