import pandas as pd
import streamlit as st

from lab import db
from lab.charts import loglog_scatter
from lab.registry import entry
from lab.ui import empty_state, page_header, provenance_footer

ENTRY = entry("email-eu-core-stats")
CODE_PATHS = (
    "pipelines/ingest/snap.py",
    "pipelines/experiments/network_science/graph_stats.py",
)

page_header(ENTRY)

st.header("Purpose")
st.write(
    "A first look at a real communication network: how big it is, how connected it is, "
    "and how unevenly activity is spread across people. It also serves as the template "
    "for every other entry in the lab."
)

st.header("Method")
st.markdown(
    "- **Data:** SNAP's *email-Eu-core* edge list. An edge *u → v* means *u* sent at least "
    "one email to *v*. Node IDs are anonymised integers.\n"
    "- **Graph:** directed, with self-loops kept for degree counts.\n"
    "- **Clustering:** computed on the undirected simple projection (self-loops removed).\n"
    "- **Components:** weakly connected (ignoring direction) and strongly connected "
    "(following direction).\n"
    "- Computed offline with NetworkX; the page only reads the stored results."
)

run = db.latest_run(ENTRY.slug)
results = db.results(run.id) if run else pd.DataFrame()
if run is None or results.empty:
    st.header("Results")
    empty_state("No published results yet. They appear here once the pipeline has run.")
    st.stop()

scalars = results[results["dims"].map(len) == 0].set_index("metric")["value"]
dist = results[results["metric"] == "degree_count"]
dist = pd.DataFrame(
    {
        "kind": dist["dims"].map(lambda d: d["kind"]),
        "degree": dist["dims"].map(lambda d: int(d["degree"])),
        "nodes": dist["value"].astype(int),
    }
)


def value(metric: str) -> float:
    """A scalar result, or NaN if this run did not record it."""
    return float(scalars.get(metric, float("nan")))


st.header("Results")
row1 = st.columns(4)
row1[0].metric("Nodes", f"{value('nodes'):,.0f}")
row1[1].metric("Edges", f"{value('edges'):,.0f}")
row1[2].metric("Density", f"{value('density'):.4f}")
row1[3].metric("Reciprocity", f"{value('reciprocity'):.1%}")
row2 = st.columns(4)
row2[0].metric("Largest weak component", f"{value('largest_wcc_fraction'):.1%}")
row2[1].metric("Largest strong component", f"{value('largest_scc_fraction'):.1%}")
row2[2].metric("Average clustering", f"{value('average_clustering'):.3f}")
row2[3].metric("Transitivity", f"{value('transitivity'):.3f}")

st.subheader("Degree distribution")
kind = st.segmented_control(
    "Degree", ["total", "in", "out"], default="total", required=True, key="degree_kind"
)
selected = dist[dist["kind"] == kind]
st.altair_chart(
    loglog_scatter(
        selected, x="degree", y="nodes", x_title=f"{kind.capitalize()} degree", y_title="Nodes"
    ),
    width="stretch",
)
zero = int(selected.loc[selected["degree"] == 0, "nodes"].sum())
st.caption(
    f"Log-log axes. {zero} node(s) with {kind} degree 0 are not shown." if zero else "Log-log axes."
)
with st.expander("Data"):
    st.dataframe(selected, hide_index=True, width="stretch")

st.header("Takeaways")
mean_total = 2 * value("mean_in_degree")
st.markdown(
    f"- **One dominant core.** {value('largest_scc_fraction'):.0%} of nodes belong to a single "
    "strongly connected component: within it, a message can travel from anyone to anyone "
    "along directed links.\n"
    f"- **Mostly two-way.** {value('reciprocity'):.0%} of directed links are reciprocated.\n"
    f"- **Clustered.** Average clustering is {value('average_clustering'):.2f}, much higher than "
    f"a random graph with this density ({value('density'):.3f}) would show: people's contacts "
    "tend to email each other too.\n"
    f"- **Uneven activity.** The busiest node has total degree {value('max_total_degree'):,.0f}, "
    f"about {value('max_total_degree') / mean_total:.0f}× the mean of {mean_total:.0f}; "
    "the distribution has a long right tail."
)

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
