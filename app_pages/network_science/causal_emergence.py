import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lab import db
from lab.registry import entry
from lab.ui import empty_state, page_header, provenance_footer

ENTRY = entry("causal-emergence-worm-vs-email")
CODE_PATHS = (
    "pipelines/ingest/snap.py",
    "pipelines/common/markov.py",
    "pipelines/experiments/network_science/causal_emergence.py",
)
# Categorical slots 1-2 (validated palette), one per network; gray for the null.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
NETWORKS = {"worm": ("C. elegans", BLUE), "email": ("Enron email", ORANGE)}
SHORT = {"worm": "Worm", "email": "Email"}
GROUPINGS = {
    "roles": "Degree roles (integrator, relay, broadcaster)",
    "greedy_within_roles": "Greedy search, merges within a role",
    "greedy": "Greedy search, any merge",
    "structural_equivalence": "Identical contacts",
    "louvain": "Louvain communities",
}
# The grouping each network's headline comparison uses: its best macro-nodes.
BEST = {"worm": "greedy", "email": "structural_equivalence"}
ROLES = ("integrator", "relay", "broadcaster")

page_header(ENTRY)

st.header("Purpose")
st.write(
    "Is a network sometimes better described by groups of nodes than by the nodes "
    "themselves? Klein and Hoel (2020) made that question measurable. Treat the network as "
    "a random walk, measure how much each node's position tells you about where the walk "
    "goes next (*effective information*, EI), and check whether merging nodes into "
    "*macro-nodes* raises it. When it does, the macro scale is the more informative "
    "description: *causal emergence*. They report two patterns across many real networks. "
    "Biological networks tend to show more causal emergence than technological ones, and "
    "the networks that gain most are *degenerate*, where many nodes send to the same "
    "places. I test both on two networks from earlier entries: the worm's frontal "
    "connectome and Enron's email network."
)

st.header("Method")
st.markdown(
    "- **Data:** SNAP's *C. elegans* frontal connectome (131 neurons, 764 directed "
    "synapses) and Enron email network (the giant component: 33,696 addresses). Both are "
    "unweighted, and SNAP collapsed email direction, so the email walk is symmetric.\n"
    "- **The chain:** from each node the walk moves to one of its out-neighbours, chosen "
    "evenly. Neurons with no outgoing synapse (7) have no next step and are left out.\n"
    "- **Effective information:** EI = H(⟨W_i⟩) − ⟨H(W_i)⟩, in bits: the entropy of the "
    "average row minus the average row entropy. It splits into *determinism* (how focused "
    "each node's outputs are) minus *degeneracy* (how much different nodes send to the same "
    "places). *Effectiveness* is EI divided by its maximum, log₂ of the number of nodes.\n"
    "- **Macro-nodes:** a macro-node's outputs are the average of its members' outputs; "
    "links inside it become self-loops. For the worm I merge the three degree roles from "
    "the *C. elegans* entry (in/(in+out) above 0.7 integrator, below 0.3 broadcaster), "
    "and I run a greedy search that repeatedly makes the merge that most raises EI, once "
    "only within a role and once with any merge allowed. For email (too large for a "
    "greedy search) I merge addresses with identical contacts, the purest form of "
    "degeneracy, and the Louvain communities from the Enron entry.\n"
    "- **Two controls:** random groupings with the same group sizes, and rewired versions "
    "of each network (20 for the worm, 10 for email) that keep every node's in- and "
    "out-degree but shuffle who connects to whom. Because degrees are kept, each node's "
    "row entropy and so the determinism are unchanged: any EI difference from the rewired "
    "networks is degeneracy.\n"
    "- **EI, not effectiveness:** causal emergence is defined as a gain in EI. "
    "Effectiveness divides by a maximum that shrinks as nodes merge, so it can rise while "
    "EI falls (see *The effectiveness trap*)."
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


def value(metric: str, **dims: object) -> float:
    """The value of ``metric`` whose dims match ``dims``, or NaN if this run lacks it."""
    rows = results[results["metric"] == metric]
    for k, v in dims.items():
        rows = rows[rows["dims"].map(lambda x, k=k, v=v: x.get(k) == v)]
    return float(rows["value"].iloc[0]) if len(rows) else float("nan")


def null_values(metric: str, network: str, grouping: str | None = None) -> pd.Series:
    """One value per rewired network."""
    rows = series(metric, "network", "grouping")
    rows = rows[rows["network"] == network]
    if grouping is not None:
        rows = rows[rows["grouping"] == grouping]
    return rows["value"].dropna()


def beats(observed: float, null: pd.Series) -> int:
    """How many rewired networks the observed value exceeds."""
    return int((null < observed).sum())


def degeneracy_share(network: str, deg: float) -> float:
    """Degeneracy as a share of its maximum, log2 of the nodes with outputs (EI/effectiveness)."""
    eff = value("micro_effectiveness", network=network)
    return deg * eff / value("micro_ei", network=network) if eff else float("nan")


st.header("Results")
cols = st.columns(4)
for col, network in zip(cols[:2], NETWORKS, strict=True):
    name = SHORT[network]
    ce = value("grouping_ce", network=network, grouping=BEST[network])
    null = null_values("null_ce", network, BEST[network])
    col.metric(
        f"{name}: emergence",
        f"{ce:+.2f} bits",
        f"rewired {null.mean():+.2f}" if len(null) else None,
        delta_color="off",
        delta_arrow="off",
        help="Causal emergence: EI of the best macro-nodes found minus the micro EI. "
        "Below: the mean over rewired networks with the same degrees.",
    )
for col, network in zip(cols[2:], NETWORKS, strict=True):
    name = SHORT[network]
    deg = value("micro_degeneracy", network=network)
    null = null_values("null_degeneracy", network)
    col.metric(
        f"{name}: degeneracy",
        f"{deg:.2f} bits",
        f"rewired {null.mean():.2f}" if len(null) else None,
        delta_color="off",
        delta_arrow="off",
        help="How much different nodes send to the same places; EI = determinism − degeneracy.",
    )

chance, degeneracy, groupings, trap, merges = st.tabs(
    [
        "Macro-nodes vs chance",
        "Degeneracy",
        "Every grouping",
        "The effectiveness trap",
        "Who merges",
    ]  # fmt: skip
)


def strip(metric_obs: dict[str, float], nulls: dict[str, pd.Series], x_title: str) -> go.Figure:
    """Observed value per network (diamond) against its rewired networks (gray dots)."""
    fig = go.Figure()
    labels = {n: NETWORKS[n][0] for n in NETWORKS}
    first = True
    for network, null in nulls.items():
        fig.add_trace(
            go.Scatter(
                x=null,
                y=[labels[network]] * len(null),
                mode="markers",
                marker={
                    "color": GRAY,
                    "size": 9,
                    "opacity": 0.6,
                    "line": {"width": 2, "color": "rgba(0,0,0,0)"},
                },  # fmt: skip
                name="Rewired networks",
                legendgroup="null",
                showlegend=first,
                hovertemplate="Rewired network: %{x:.3f}<extra></extra>",
            )
        )
        first = False
    for network, observed in metric_obs.items():
        name, color = NETWORKS[network]
        fig.add_trace(
            go.Scatter(
                x=[observed],
                y=[name],
                mode="markers",
                marker={
                    "color": color,
                    "size": 16,
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "white"},
                },  # fmt: skip
                name=name,
                hovertemplate=f"{name}: %{{x:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title=x_title,
        legend={"orientation": "h", "y": -0.35},
        yaxis={"categoryorder": "array", "categoryarray": [labels[n] for n in NETWORKS][::-1]},
    )
    return fig


with chance:
    obs = {n: value("grouping_ce", network=n, grouping=BEST[n]) for n in NETWORKS}
    nulls = {n: null_values("null_ce", n, BEST[n]) for n in NETWORKS}
    fig = strip(obs, nulls, "Causal emergence (bits): best macro-nodes EI − micro EI")
    fig.add_vline(x=0, line={"color": GRAY, "width": 1, "dash": "dot"})
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each diamond is the real network's best macro-scale: for the worm, the greedy "
        "search with any merge allowed; for email, merging addresses with identical "
        "contacts. Gray dots are the same procedure on rewired networks with the same degrees."
    )

with degeneracy:
    obs = {n: value("micro_degeneracy", network=n) for n in NETWORKS}
    nulls = {n: null_values("null_degeneracy", n) for n in NETWORKS}
    st.plotly_chart(strip(obs, nulls, "Degeneracy (bits)"), width="stretch")
    rows = []
    for network, (name, _) in NETWORKS.items():
        null_ei = null_values("null_ei", network)
        rows.append(
            {
                "Network": name,
                "Micro EI": value("micro_ei", network=network),
                "Rewired EI (mean)": null_ei.mean() if len(null_ei) else float("nan"),
                "Determinism": value("micro_determinism", network=network),
                "Rewired determinism (mean)": null_values("null_determinism", network).mean(),
                "Degeneracy": obs[network],
                "Rewired degeneracy (mean)": nulls[network].mean(),
                "Degeneracy, share of max": degeneracy_share(network, obs[network]),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in rows[0] if c != "Network"
        },  # fmt: skip
    )
    st.caption(
        "All values in bits except the share. Rewiring keeps degrees, so determinism is "
        "identical by construction and the EI gap is entirely degeneracy. Degeneracy's "
        "maximum is log₂ of the nodes with outputs (about 7 bits for the worm, 15 for email), "
        "so the share compares the two networks on one scale."
    )

with groupings:
    table = []
    for network, (name, _) in NETWORKS.items():
        micro = value("micro_ei", network=network)
        table.append({"Network": name, "Grouping": "Individual nodes",
                      "Macro-nodes": value("nodes", network=network), "EI": micro,
                      "Causal emergence": 0.0,
                      "Effectiveness": value("micro_effectiveness", network=network)})  # fmt: skip
        for grouping, label in GROUPINGS.items():
            ei = value("grouping_ei", network=network, grouping=grouping)
            if math.isnan(ei):
                continue
            null = null_values("null_ce", network, grouping)
            table.append(
                {
                    "Network": name,
                    "Grouping": label,
                    "Macro-nodes": value("grouping_groups", network=network, grouping=grouping),
                    "EI": ei,
                    "Causal emergence": value("grouping_ce", network=network, grouping=grouping),
                    "Effectiveness": value(
                        "grouping_effectiveness", network=network, grouping=grouping
                    ),
                    "Random same-size EI": value(
                        "random_ei_mean", network=network, grouping=grouping
                    ),
                    "Beats random": value("random_beaten", network=network, grouping=grouping),
                    "Rewired CE (mean)": null.mean() if len(null) else None,
                }
            )
    st.dataframe(
        pd.DataFrame(table),
        hide_index=True,
        column_config={
            "Macro-nodes": st.column_config.NumberColumn(format="localized"),
            "EI": st.column_config.NumberColumn(format="%.3f"),
            "Causal emergence": st.column_config.NumberColumn(format="%+.3f"),
            "Effectiveness": st.column_config.NumberColumn(format="%.3f"),
            "Random same-size EI": st.column_config.NumberColumn(format="%.3f"),
            "Beats random": st.column_config.NumberColumn(
                format="percent", help="Share of random same-size groupings with lower EI."
            ),
            "Rewired CE (mean)": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    st.caption(
        "EI and causal emergence in bits. Random same-size groupings: 1,000 for the worm, "
        "20 (identical contacts) and 100 (Louvain) for email. Rewired CE repeats the same "
        "grouping procedure on each rewired network; for Louvain that means new communities."
    )

with trap:
    path = series("path_ei", "network", "groups").merge(
        series("path_effectiveness", "network", "groups"),
        on=["network", "groups"],
        suffixes=("_ei", "_eff"),
    )
    path = path[path["network"] == "worm"].sort_values("groups", ascending=False)
    if path.empty:
        empty_state("This run recorded no merge path.")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        for row, column, title in ((1, "value_ei", "EI (bits)"),
                                   (2, "value_eff", "Effectiveness")):  # fmt: skip
            fig.add_trace(
                go.Scatter(
                    x=path["groups"],
                    y=path[column],
                    mode="lines",
                    line={"color": BLUE, "width": 2},
                    name=title,
                    showlegend=False,
                    hovertemplate=f"%{{x}} macro-nodes<br>{title}: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text=title, row=row, col=1)
        fig.update_xaxes(autorange="reversed", title_text="Macro-nodes (merging left to right)",
                         row=2, col=1)  # fmt: skip
        fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(fig, width="stretch")
        peak = path.loc[path["value_eff"].idxmax()]
        start = path.iloc[0]
        st.caption(
            "The worm's greedy merge path, restricted to merges within a role, from 131 "
            f"neurons down to the three roles. Effectiveness peaks at {peak['value_eff']:.2f} "
            f"with {peak['groups']:.0f} macro-nodes, above the micro scale's "
            f"{start['value_eff']:.2f}, while EI there is {peak['value_ei']:.2f} bits against "
            f"{start['value_ei']:.2f}. The rise comes from the shrinking maximum, not from "
            "more information."
        )

with merges:
    nodes = series("macro_node", "network", "rank", "members", *ROLES)
    st.subheader("C. elegans: the greedy search's macro-nodes")
    if nodes.empty:
        empty_state("The greedy search merged no neurons.")
    else:
        st.dataframe(
            nodes.rename(
                columns={
                    "value": "Neurons",
                    "members": "Members",
                    "integrator": "Integrators",
                    "relay": "Relays",
                    "broadcaster": "Broadcasters",
                }
            )[["Neurons", "Integrators", "Relays", "Broadcasters", "Members"]],  # fmt: skip
            hide_index=True,
        )
    st.subheader("Enron: addresses with identical contacts")
    classes = series("equivalence_class", "rank", "contacts")
    merged = value("grouping_merged", network="email", grouping="structural_equivalence")
    null_merged = null_values("null_merged", "email", "structural_equivalence")
    null_largest = null_values("null_largest_class", "email")
    if classes.empty:
        empty_state("No addresses share identical contacts.")
    else:
        one = int((classes["contacts"] == 1).sum())
        st.markdown(
            f"- {merged:,.0f} addresses fall into groups with identical contacts"
            + (
                f" (rewired networks: {null_merged.mean():,.0f} on average)."
                if len(null_merged)
                else "."
            )  # fmt: skip
            + f"\n- The largest group has {classes['value'].max():,.0f} addresses"
            + (f" (rewired: at most {null_largest.max():,.0f})." if len(null_largest) else ".")
            + f"\n- In {one} of the {len(classes)} largest groups, every address has exactly one "
            "contact, the same person for the whole group (SNAP does not say who wrote to whom)."
        )
        st.dataframe(
            classes.rename(
                columns={"rank": "Rank", "value": "Addresses", "contacts": "Shared contacts"}
            ),  # fmt: skip
            hide_index=True,
        )

st.header("Takeaways")
worm_ce = value("grouping_ce", network="worm", grouping="greedy")
worm_null = null_values("null_ce", "worm", "greedy")
email_ce = value("grouping_ce", network="email", grouping="structural_equivalence")
email_null = null_values("null_ce", "email", "structural_equivalence")
roles_ei = value("grouping_ei", network="worm", grouping="roles")
lv_ce = value("grouping_ce", network="email", grouping="louvain")
lv_eff = value("grouping_effectiveness", network="email", grouping="louvain")
bullets = []
if not math.isnan(roles_ei):
    bullets.append(
        f"**The worm's degree roles carry almost no information.** Merged into three "
        f"macro-nodes, the 131 neurons keep {roles_ei:.3f} of "
        f"{value('micro_ei', network='worm'):.2f} bits. All three roles send to about the "
        "same mix of targets, because a role describes a neuron's balance of inputs and "
        "outputs, not where its outputs go."
    )
if len(worm_null):
    bullets.append(
        f"**No convincing causal emergence in the worm.** The best macro-nodes the greedy "
        f"search finds gain {worm_ce:+.3f} bits, and the same search on rewired networks "
        f"gains {worm_null.mean():+.3f} on average ({worm_null.min():+.3f} to "
        f"{worm_null.max():+.3f}); the real network beats {beats(worm_ce, worm_null)} of "
        f"{len(worm_null)}, so its gain is about what chance wiring with the same degrees "
        "gives."
        + (
            " The largest macro-node is mostly broadcasters, many of them head motor neurons "
            "by name (SMD, RMD, RME), a link that comes from the neuroscience literature "
            "rather than from this analysis."
            if any("RMD" in str(m) for m in series("macro_node", "members")["members"])
            else ""
        )
    )
if len(email_null):
    bullets.append(
        f"**Clear causal emergence in email, from degeneracy.** Merging addresses with "
        f"identical contacts gains {email_ce:+.3f} bits; in the rewired networks the same "
        f"merge changes EI by {email_null.min():+.3f} to {email_null.max():+.3f}. Enron's "
        "email network is far more degenerate than its degrees require: thousands of "
        "addresses whose only contact is one of the same few people."
    )
excess = {}
for network in NETWORKS:
    null = null_values("null_degeneracy", network)
    if len(null):
        deg = value("micro_degeneracy", network=network)
        excess[network] = (deg - null.mean(), beats(deg, null), len(null))
if len(excess) == 2:
    (w_ex, w_b, w_n), (e_ex, e_b, e_n) = excess["worm"], excess["email"]
    bullets.append(
        f"**Both networks are more degenerate than chance; only email's redundancy can be "
        f"merged.** Against its rewired versions the worm has {w_ex:+.2f} bits of extra "
        f"degeneracy (above {w_b} of {w_n}) and email {e_ex:+.2f} (above {e_b} of {e_n}). "
        "The worm's overlap is spread thinly across neurons whose targets partly coincide, so "
        "no macro-node captures it; email's sits in large groups of identical addresses."
    )
if not math.isnan(lv_ce):
    bullets.append(
        f"**Communities are informative, but not macro-nodes in Klein's sense.** Louvain's "
        f"communities lose {-lv_ce:.2f} bits against individual addresses, yet raise "
        f"effectiveness to {lv_eff:.2f}. They are far better than random groupings of the "
        "same sizes, and the rise in effectiveness is the trap shown above."
    )
st.markdown("\n".join(f"- {b}" for b in bullets))

email_wins = len(email_null) and email_ce > email_null.max()
worm_wins = len(worm_null) and worm_ce > worm_null.max()
st.subheader("Against Klein and Hoel's patterns")
if email_wins and not worm_wins:
    st.write(
        "My pair fits their second pattern and runs against their first. The network that "
        "gains from coarse-graining is the degenerate one, as they predict, but it is the "
        "email network, not the connectome. I don't read this as a counterexample. Two "
        "networks can't test a trend across domains, Enron is a social network rather than a "
        "technological one, and this connectome is an unweighted, 131-neuron subset of the "
        "worm. Weighted synapse counts or the whole nervous system could change the worm's "
        "result. What the pair does show is the mechanism: macro-nodes beat micro-nodes when "
        "the micro scale is full of redundant nodes, and the redundancy has to exceed what "
        "the degrees alone produce."
    )
else:
    st.write(
        "Two networks can't test a trend across domains, so I compare this run with Klein "
        "and Hoel's patterns only through the table above: which network is more degenerate "
        "than its rewired versions, and which gains from macro-nodes beyond them."
    )

st.subheader("Limitations")
st.markdown(
    "- Both networks are unweighted, and the email network is undirected.\n"
    "- The greedy search can miss better groupings, and for email I tested two hand-picked "
    "groupings rather than searching. The email gain is a lower bound.\n"
    "- 20 and 10 rewired networks give the null's spread, not precise tail probabilities.\n"
    "- Macro-nodes average members' outputs evenly, matching EI's even intervention. "
    "Weighting by how often the walk visits each node gave the same conclusions when I "
    "explored this."
)

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
