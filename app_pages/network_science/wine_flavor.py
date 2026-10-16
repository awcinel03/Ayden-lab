import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import db, storage
from lab.registry import entry
from lab.ui import REPO_URL, empty_state, page_header, provenance_footer

ENTRY = entry("wine-flavor-network")
CODE_PATHS = (
    "pipelines/ingest/wine_flavor.py",
    "pipelines/ingest/wine_flavor_lexicon_v2.json",
    "pipelines/ingest/wine_flavor_lexicon_v1.json",
    "pipelines/common/network_metrics.py",
    "pipelines/experiments/network_science/wine_flavor.py",
)
# Categorical slots 1-3 (validated palette), assigned to communities by size rank.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COMMUNITY_COLORS = {1: BLUE, 2: ORANGE, 3: AQUA}
OTHER_COLOR = "#8c8c8c"
LEXICON_V2_URL = f"{REPO_URL}/blob/main/pipelines/ingest/wine_flavor_lexicon_v2.json"
# Marker terms for the red/white reading (checked against each partition, never assumed).
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
    "*Conspiracies, Connectomes, and Cabernet*. Since writing it I have corrected how terms "
    "are matched in the reviews; the results below use the corrected rule, and a section "
    "near the end shows what changed from my original analysis."
)

st.header("Method")
st.markdown(
    "- **Data:** the Kaggle *Wine Reviews* dataset (winemag-data-130k-v2, reviews scraped from "
    "Wine Enthusiast). The review text belongs to Wine Enthusiast, so it never leaves my "
    "machine: the pipeline publishes only derived counts (how many reviews mention each term, "
    "each pair of terms, and how many each matching rule alone would catch). "
    "No review text is stored or shown here.\n"
    "- **Lexicon:** the same 116 descriptors as my notebook, taken from the UC Davis Wine "
    "Aroma Wheel (fruit, floral, spice, earth, oak, plus structural words such as *acid*, "
    "*acidity*, *tannin* and *fresh*). Version 2 of the lexicon adds, for every term, an "
    f"explicit list of accepted word forms ([wine_flavor_lexicon_v2.json]({LEXICON_V2_URL})): "
    "the term plus plurals and derivations of the same flavour, e.g. *tar, tarry*; *jam, jams, "
    "jammy, jamminess*; *herbs, herb*; *tannin, tannins, tannic*. Words that merely contain "
    "the term (*tart*, *pineapple*, *breadth*), other lexicon terms (*acidity* is not a form "
    "of *acid*; *smoky* is not a form of *smoke*), negations (*unoaked*) and typos are not "
    "forms. Each form belongs to one term only.\n"
    "- **Matching (v2):** each review is lower-cased and split into words with the regular "
    "expression `\\w+` (letters, digits and underscores; hyphens and apostrophes split words, "
    "so *tar-like* gives *tar*). A review mentions a term if any of its forms is one of those "
    "words. Reviews with fewer than two terms are skipped, as in my notebook. My original "
    "analysis matched substrings instead; see the last section.\n"
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


def sides(communities: pd.Series) -> dict[str, str]:
    """Map each term to "red", "white" or "community N", if the markers split cleanly."""
    red = {communities[t] for t in RED_MARKERS if t in communities.index}
    white = {communities[t] for t in WHITE_MARKERS if t in communities.index}
    if len(red) != 1 or len(white) != 1 or red == white:
        return {t: f"community {c:.0f}" for t, c in communities.items()}
    r, w = red.pop(), white.pop()
    names = {r: "red", w: "white"}
    return {t: names.get(c, f"community {c:.0f}") for t, c in communities.items()}


comm_v2 = series("term_community", "term").set_index("term")["value"]
comm_v1 = series("v1_term_community", "term").set_index("term")["value"]
side_v2, side_v1 = sides(comm_v2), sides(comm_v1)
split_v2 = "red" in side_v2.values() and "white" in side_v2.values()
split_v1 = "red" in side_v1.values() and "white" in side_v1.values()

st.header("What changed from my original analysis")
v1_terms = series("v1_term_reviews", "term", "degree")
if v1_terms.empty:
    empty_state("This run has no comparison with my original matching.")
else:
    rev1 = v1_terms.set_index("term")["value"]
    rev2 = terms.set_index("term")["value"]
    removed = series("substring_only_reviews", "term").set_index("term")["value"]
    added = series("forms_only_reviews", "term").set_index("term")["value"]
    st.write(
        "My original notebook counted a term whenever its letters appeared anywhere in a "
        "review. That over-counts: *tar* matched inside *tart*, *nectarine* and *star*; *acid* "
        "inside *acidity*; *apple* inside *pineapple*; *honey* inside *honeysuckle* and "
        "*honeydew*. It also under-counts, because a review that says *herb* or *tannic* never "
        "contains the letters *herbs* or *tannin*. Version 2 fixes both with explicit word forms "
        "matched as whole words. The two effects are separate: false matches removed, and "
        "missed forms added."
    )

    def change(term: str) -> str:
        """``v1 → v2`` review counts for one term."""
        return f"{rev1.get(term, 0):,.0f} → {rev2.get(term, 0):,.0f}"

    points = []
    if "tar" in rev1.index:
        points.append(
            f"**tar:** {change('tar')} reviews; {removed.get('tar', 0):,.0f} reviews matched "
            "*tar* only inside other words."
        )
    v1_pairs = series("v1_top_pair", "rank", "term_a", "term_b")
    if not v1_pairs.empty and "acidity" in rev1.index:
        top1 = v1_pairs.iloc[0]
        same = {top1.term_a, top1.term_b} == {"acid", "acidity"}
        if same and top1.value == rev1["acidity"]:
            points.append(
                f"**acid:** {change('acid')} reviews. My original strongest pair, acid + "
                f"acidity ({top1.value:,.0f} reviews), is exactly the number of reviews "
                "mentioning *acidity*: every one also counted as *acid*, so the pair was "
                "mostly one word counted twice."
            )
        else:
            points.append(f"**acid:** {change('acid')} reviews.")
    for term, inside in (("apple", "pineapple"), ("honey", "honeysuckle and honeydew")):
        if term in rev1.index:
            points.append(
                f"**{term}:** {change(term)} reviews, no longer counting *{inside}* "
                f"({removed.get(term, 0):,.0f} reviews matched only that way)."
            )
    gains = added.sort_values(ascending=False).head(3)
    if len(gains) and gains.iloc[0] > 0:
        points.append(
            "**Missed forms:** the largest additions are "
            + ", ".join(f"{t} ({change(t)}; +{v:,.0f} via its new forms)" for t, v in gains.items())
            + "."
        )
    st.markdown("\n".join(f"- {x}" for x in points))

    with st.container(horizontal=True):
        st.metric("False matches removed", f"{removed.sum():,.0f}", border=True)
        st.metric("Missed forms added", f"{added.sum():,.0f}", border=True)
        both = pd.DataFrame({"v1": rev1, "v2": rev2}).fillna(0)
        rel = (both["v2"] - both["v1"]) / both["v1"].where(both["v1"] > 0)
        st.metric("Terms down more than 10%", f"{(rel < -0.1).sum():,.0f}", border=True)
        st.metric("Terms up more than 10%", f"{(rel > 0.1).sum():,.0f}", border=True)
    st.caption(
        "Removed and added count review–term matches over all reviews: a review matched by "
        "only one of the two rules. The term counts below include only reviews with at least "
        "two terms, so they don't add up exactly."
    )
    moved = both.assign(change=both["v2"] - both["v1"])
    moved = moved.reindex(moved["change"].abs().sort_values(ascending=False).index).head(12)
    st.dataframe(
        pd.DataFrame(
            {
                "Term": moved.index,
                "Reviews (original)": moved["v1"].astype(int).to_numpy(),
                "Reviews (v2)": moved["v2"].astype(int).to_numpy(),
                "Matched only inside other words": removed.reindex(moved.index)
                .fillna(0)
                .astype(int)
                .to_numpy(),
                "Matched only by added forms": added.reindex(moved.index)
                .fillna(0)
                .astype(int)
                .to_numpy(),
            }
        ),
        hide_index=True,
        width="stretch",
    )

    def leader(df: pd.DataFrame, col: str) -> str:
        """The term(s) with the highest ``col``."""
        top = df[df[col] == df[col].max()]
        return ", ".join(top["term"]) + f" ({df[col].max():,.0f})"

    def pair(metric: str) -> str:
        """The strongest pair of a run's ``top_pair`` series."""
        p = series(metric, "rank", "term_a", "term_b")
        return (
            f"{p.iloc[0].term_a} + {p.iloc[0].term_b} ({p.iloc[0].value:,.0f})" if len(p) else "–"
        )

    def at_v1(strategy: str, step: int) -> float:
        """A v1 giant fraction at ``step``, or NaN."""
        c = series("v1_giant_fraction", "strategy", "step")
        c = c[(c["strategy"] == strategy) & (c["step"] == step)]
        return float(c["value"].iloc[0]) if len(c) else float("nan")

    sizes_v1 = series("v1_community_size", "rank")["value"]
    compare = pd.DataFrame(
        [
            (
                "Edges (pairs in ≥ 10 reviews)",
                f"{value('v1_edges'):,.0f}",
                f"{value('edges'):,.0f}",
            ),
            ("Average degree", f"{value('v1_avg_degree'):.2f}", f"{value('avg_degree'):.2f}"),
            ("Density", f"{value('v1_density'):.3f}", f"{value('density'):.3f}"),
            (
                "Average clustering",
                f"{value('v1_avg_clustering'):.3f}",
                f"{value('avg_clustering'):.3f}",
            ),
            ("Most frequent term", leader(v1_terms, "value"), leader(terms, "value")),
            ("Highest degree", leader(v1_terms, "degree"), leader(terms, "degree")),
            ("Strongest pair (reviews)", pair("v1_top_pair"), pair("top_pair")),
            (
                "Louvain communities (seed 42), sizes",
                ", ".join(f"{v:.0f}" for v in sizes_v1),
                ", ".join(f"{v:.0f}" for v in sizes["value"]),
            ),
            ("Modularity", f"{value('v1_modularity'):.3f}", f"{value('modularity'):.3f}"),
            (
                "Largest component after 25 removals (targeted / random)",
                f"{at_v1('targeted', 25):.3f} / {at_v1('random', 25):.3f}",
                f"{t25:.3f} / {r25:.3f}",
            ),
            ("Community of tar", side_v1.get("tar", "–"), side_v2.get("tar", "–")),
        ],
        columns=["", "Original (substring)", "v2 (word forms)"],
    )
    st.dataframe(compare, hide_index=True, width="stretch")

    st.subheader("What that does to the conclusions")
    lines = []
    if "tar" in side_v1 and "tar" in side_v2 and side_v1["tar"] != side_v2["tar"]:
        lines.append(
            f"**tar moves from the {side_v1['tar']} community to the {side_v2['tar']} one.** "
            "In my original analysis "
            f"{removed.get('tar', 0) / rev1['tar']:.0%} of its matches came from other words "
            "such as *tart*, *nectarine* and *star*, which pulled it towards the white-wine "
            "terms. I read the move as tar going back to where tasters use it, for red wines."
        )
    if split_v1 and split_v2:
        lines.append(
            "**The red/white split holds.** In both versions "
            + ", ".join(RED_MARKERS)
            + " share one community and "
            + ", ".join(WHITE_MARKERS)
            + " another."
        )
    elif split_v1 and not split_v2:
        lines.append("**The red/white split no longer holds** with the corrected matching.")
    moves: dict[tuple[str, str], list[str]] = {}
    for t in side_v2:
        before, after = side_v1.get(t), side_v2[t]
        if t == "tar" or before is None or before == after:
            continue
        if before.startswith("community") and after.startswith("community"):
            continue  # an unlabelled group renumbered, not a move
        moves.setdefault((before, after), []).append(t)
    if moves:
        size_of = comm_v2.value_counts()
        lines.append(
            "**Other terms changed community:** "
            + "; ".join(
                f"{', '.join(ts)} ({a} → {b}"
                + (
                    f", a separate group of {size_of[float(b.split()[1])]:.0f} terms"
                    if b.startswith("community")
                    else ""
                )
                + ")"
                for (a, b), ts in sorted(moves.items(), key=lambda x: -len(x[1]))
            )
            + "."
        )
    if value("v1_edges") > value("edges"):
        lines.append(
            "**Slightly less dense, just as near-complete:** "
            f"{value('v1_edges') - value('edges'):,.0f} fewer edges and average degree "
            f"{value('v1_avg_degree'):.2f} → "
            f"{value('avg_degree'):.2f}. Robustness after 25 removals is "
            + (
                "unchanged."
                if (round(at_v1("targeted", 25), 3), round(at_v1("random", 25), 3))
                == (round(t25, 3), round(r25, 3))
                else f"{at_v1('targeted', 25):.3f} / {at_v1('random', 25):.3f} → "
                f"{t25:.3f} / {r25:.3f} (targeted / random)."
            )
        )
    st.markdown("\n".join(f"- {x}" for x in lines))

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
if not comm_v2.empty:
    sentence = (
        f"**Communities.** Louvain finds {value('communities'):.0f} communities with modularity "
        f"{value('modularity'):.3f}, sizes " + ", ".join(f"{v:.0f}" for v in sizes["value"]) + "."
    )
    if split_v2:
        sentence += (
            f" {', '.join(RED_MARKERS)} all fall in one, and {', '.join(WHITE_MARKERS)} in "
            "another. I read this as the red/white divide, recovered from co-occurrence alone; "
            "the algorithm knows nothing about wine."
        )
    reviews = terms.set_index("term")["value"]
    others = [
        (c, grp.index.tolist())
        for c, grp in comm_v2.groupby(comm_v2)
        if len(grp) > 1 and not any(side_v2[t] in ("red", "white") for t in grp.index)
    ]
    for _, members in others:
        lead = sorted(members, key=lambda t: -reviews.get(t, 0))
        sentence += (
            f" A further community of {len(members)} terms ({', '.join(lead[:6])}"
            + ("…" if len(lead) > 6 else "")
            + ") stands apart"
            + ("; I read it as the vocabulary of barrel ageing" if "oak" in members else "")
            + "."
        )
    sweep_n = series("sweep_communities", "seed")["value"]
    if len(sweep_n) and others:
        same = int((sweep_n == value("communities")).sum())
        sentence += (
            f" {same} of {len(sweep_n)} other seeds also find {value('communities'):.0f} "
            "communities; the rest find "
            + " or ".join(sorted({f"{v:.0f}" for v in sweep_n if v != value("communities")}))
            + ", so the smaller group is not found every time."
            if same < len(sweep_n)
            else ""
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
