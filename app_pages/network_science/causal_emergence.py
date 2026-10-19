import math
from pathlib import Path

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
    "pipelines/ingest/statsbomb.py",
    "pipelines/common/markov.py",
    "pipelines/experiments/network_science/football.py",
    "pipelines/experiments/network_science/causal_emergence.py",
)
# Categorical slots 1-3 (validated palette), one per network; gray for the null.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
NETWORKS = {
    "worm": ("C. elegans", BLUE),
    "email": ("Enron email", ORANGE),
    "football": ("Arsenal passing", AQUA),
}
SHORT = {"worm": "Worm", "email": "Email", "football": "Arsenal"}
# StatsBomb's licence requires its logo on any published analysis of the data.
ASSETS = Path(__file__).resolve().parents[2] / "lab" / "assets"
GROUPINGS = {
    "roles": "Degree roles (integrator, relay, broadcaster)",
    "lines": "Positional lines (keeper, defence, midfield, attack)",
    "greedy_within_roles": "Greedy search, merges within a role",
    "greedy": "Greedy search, any merge",
    "structural_equivalence": "Identical contacts",
    "louvain": "Louvain communities",
}
# Arsenal's roles are positional (side x line), not the worm's degree roles.
FOOTBALL_LABELS = {"roles": "Positional roles (side × line)"}
# The grouping each network's headline comparison uses: its best macro-nodes.
BEST = {"worm": "greedy", "email": "structural_equivalence", "football": "greedy"}
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
    "places. I test both on three networks from earlier entries: the worm's frontal "
    "connectome, Enron's email network and Arsenal's passing in the 2003/04 season."
)

st.header("Method")
st.markdown(
    "- **Data:** SNAP's *C. elegans* frontal connectome (131 neurons, 764 directed "
    "synapses) and Enron email network (the giant component: 33,696 addresses). Both are "
    "unweighted, and SNAP collapsed email direction, so the email walk is symmetric. "
    "Arsenal's chains come from StatsBomb events for all 38 league matches, as in the "
    "football entry: the starting XI from kick-off to the first substitution or red card "
    "(a median of 318 completed passes), with passes counted, so each player's row is the "
    "share of their passes to each teammate. Season values for Arsenal are medians over "
    "matches.\n"
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
    "degeneracy, and the Louvain communities from the Enron entry. For Arsenal I merge "
    "players into positional roles and lines and run the greedy search. I weight members "
    "evenly here, where the football entry weights them by passes made, so its Arsenal "
    "numbers differ slightly from these.\n"
    "- **Two controls:** random groupings with the same group sizes, and rewired versions "
    "of each network (20 for the worm, 10 for email) that keep every node's in- and "
    "out-degree but shuffle who connects to whom. Because degrees are kept, each node's "
    "row entropy and so the determinism are unchanged: any EI difference from the rewired "
    "networks is degeneracy. For Arsenal I use the football entry's null: 500 random "
    "versions of each match that keep every player's passes made and received (100 for the "
    "greedy search). Passes are counted, so that null changes each player's spread too, "
    "and determinism can move as well as degeneracy.\n"
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
    """Degeneracy as a share of its maximum, log2 of the nodes with outputs."""
    if network == "football":
        return deg / math.log2(value("nodes", network=network))
    eff = value("micro_effectiveness", network=network)
    return deg * eff / value("micro_ei", network=network) if eff else float("nan")


def per_match(*metrics: str) -> pd.DataFrame:
    """Arsenal's per-match values of ``metrics``, one row per match."""
    out = None
    for metric in metrics:
        s = series(f"match_{metric}", "match").rename(columns={"value": metric})
        out = s if out is None else out.merge(s, on="match")
    return out if out is not None else pd.DataFrame()


def null_reference(network: str, metric: str, grouping: str | None = None) -> float:
    """The null's typical value: rewired mean (worm, email) or median of match means."""
    if network == "football":
        name = {"ce": "null_ce_greedy"}.get(metric, f"null_{metric}")
        col = per_match(name)
        return float(col[name].median()) if not col.empty else float("nan")
    null = null_values(f"null_{metric}", network, grouping)
    return float(null.mean()) if len(null) else float("nan")


def excess(network: str, kind: str) -> tuple[pd.Series, pd.Series]:
    """Observed minus the null's mean, and the null draws' own spread around their mean.

    ``kind`` is "ce" (bits; the network's best macro-nodes) or "degeneracy" (share of its
    maximum). Arsenal has one observed value per match and no pooled null draws.
    """
    if network == "football":
        if kind == "ce":
            m = per_match("greedy_ce", "null_ce_greedy")
            obs = m["greedy_ce"] - m["null_ce_greedy"] if not m.empty else pd.Series()
        else:
            m = per_match("degeneracy", "null_degeneracy", "ei", "effectiveness")
            top = m["ei"] / m["effectiveness"] if not m.empty else pd.Series()
            obs = (m["degeneracy"] - m["null_degeneracy"]) / top if not m.empty else pd.Series()
        return obs.dropna(), pd.Series(dtype=float)
    if kind == "ce":
        observed = value("grouping_ce", network=network, grouping=BEST[network])
        null = null_values("null_ce", network, BEST[network])
        scale = 1.0
    else:
        observed = value("micro_degeneracy", network=network)
        null = null_values("null_degeneracy", network)
        scale = degeneracy_share(network, 1.0)
    if not len(null):
        return pd.Series(dtype=float), null
    return pd.Series([(observed - null.mean()) * scale]), (null - null.mean()) * scale


st.header("Results")
for col, network in zip(st.columns(3), NETWORKS, strict=True):
    name = SHORT[network]
    ce = value("grouping_ce", network=network, grouping=BEST[network])
    null_ce = null_reference(network, "ce", BEST[network])
    col.metric(
        f"{name}: emergence",
        f"{ce:+.2f} bits",
        f"null {null_ce:+.2f}" if not math.isnan(null_ce) else None,
        delta_color="off",
        delta_arrow="off",
        help="Causal emergence: EI of the best macro-nodes found minus the micro EI. Below: "
        "the same search on the null networks. Arsenal: medians over matches.",
    )
    deg = value("micro_degeneracy", network=network)
    null_deg = null_reference(network, "degeneracy")
    col.metric(
        f"{name}: degeneracy",
        f"{deg:.2f} bits",
        f"null {null_deg:.2f}" if not math.isnan(null_deg) else None,
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


def excess_chart(kind: str, x_title: str) -> go.Figure:
    """Each network's observed value minus its null's mean; gray dots are the null's spread."""
    fig = go.Figure()
    labels = [NETWORKS[n][0] for n in NETWORKS]
    first = True
    for network, (name, color) in NETWORKS.items():
        obs, spread = excess(network, kind)
        if len(spread):
            fig.add_trace(
                go.Scatter(
                    x=spread,
                    y=[name] * len(spread),
                    mode="markers",
                    marker={
                        "color": GRAY,
                        "size": 9,
                        "opacity": 0.6,
                        "line": {"width": 2, "color": "rgba(0,0,0,0)"},
                    },  # fmt: skip
                    name="Null networks",
                    legendgroup="null",
                    showlegend=first,
                    hovertemplate="Null network: %{x:+.3f}<extra></extra>",
                )
            )
            first = False
        many = len(obs) > 1
        fig.add_trace(
            go.Scatter(
                x=obs,
                y=[name] * len(obs),
                mode="markers",
                marker={
                    "color": color,
                    "size": 9 if many else 16,
                    "symbol": "circle" if many else "diamond",
                    "opacity": 0.75 if many else 1,
                    "line": {"width": 2, "color": "white"},
                },  # fmt: skip
                name=f"{name} (one dot per match)" if many else name,
                hovertemplate=f"{name}: %{{x:+.3f}}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line={"color": GRAY, "width": 1, "dash": "dot"})
    fig.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title=x_title,
        legend={"orientation": "h", "y": -0.3},
        yaxis={"categoryorder": "array", "categoryarray": labels[::-1]},
    )
    return fig


with chance:
    st.plotly_chart(
        excess_chart("ce", "Causal emergence above the null's mean (bits)"), width="stretch"
    )
    st.caption(
        "Each network's best macro-nodes (worm: the greedy search with any merge; email: "
        "addresses with identical contacts; Arsenal: the greedy search, per match), measured "
        "against the same procedure on its null networks. Zero means no more emergence than "
        "chance wiring with the same degrees gives. Gray dots show the null's own spread; for "
        "Arsenal each dot is one match compared with its own 100 null versions."
    )

with degeneracy:
    st.plotly_chart(
        excess_chart("degeneracy", "Degeneracy above the null's mean (share of its maximum)"),
        width="stretch",
    )
    rows = []
    for network, (name, _) in NETWORKS.items():
        deg = value("micro_degeneracy", network=network)
        rows.append(
            {
                "Network": name,
                "Micro EI": value("micro_ei", network=network),
                "Null EI": null_reference(network, "ei"),
                "Determinism": value("micro_determinism", network=network),
                "Null determinism": null_reference(network, "determinism"),
                "Degeneracy": deg,
                "Null degeneracy": null_reference(network, "degeneracy"),
                "Degeneracy, share of max": degeneracy_share(network, deg),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in rows[0] if c != "Network"
        },  # fmt: skip
    )
    fb = per_match("degeneracy_pct", "ei_pct")
    deg_hi = int((fb["degeneracy_pct"] > 0.975).sum()) if not fb.empty else 0
    deg_lo = int((fb["degeneracy_pct"] < 0.025).sum()) if not fb.empty else 0
    ei_hi = int((fb["ei_pct"] > 0.975).sum()) if not fb.empty else 0
    st.caption(
        "All values in bits except the share. Rewiring keeps the worm's and email's degrees, "
        "so their determinism is identical by construction and the EI gap is entirely "
        "degeneracy. Arsenal's values are medians over matches; its null changes each "
        "player's spread of passes, so determinism moves too. Degeneracy's maximum is log₂ "
        "of the nodes with outputs (about 3.5 bits for Arsenal, 7 for the worm, 15 for "
        "email), so the share compares the three on one scale."
        + (
            f" Arsenal's EI is above its null in {ei_hi} of {len(fb)} matches, while its "
            f"degeneracy is above the null in {deg_hi} and below it in {deg_lo}."
            if len(fb)
            else ""
        )
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
            if network == "football":
                label = FOOTBALL_LABELS.get(grouping, label)
                null_ce = null_reference(network, "ce") if grouping == "greedy" else None
            else:
                null = null_values("null_ce", network, grouping)
                null_ce = null.mean() if len(null) else None
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
                    "Null CE": null_ce,
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
            "Null CE": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    st.caption(
        "EI and causal emergence in bits; Arsenal's rows are medians over 38 matches. "
        "Random same-size groupings: 1,000 for the worm, 20 (identical contacts) and 100 "
        "(Louvain) for email, 100 per match for Arsenal. Null CE repeats the same grouping "
        "procedure on each null network (mean for the worm and email, median of match means "
        "for Arsenal); for Louvain that means new communities."
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
    fb_groups = per_match("greedy_groups")
    if not fb_groups.empty:
        merged_matches = int(
            (fb_groups["greedy_groups"] < value("nodes", network="football")).sum()
        )
        st.subheader("Arsenal: the greedy search's macro-nodes")
        st.write(
            f"In {merged_matches} of {len(fb_groups)} matches the greedy search found any merge "
            "that raised EI at all; in the rest it kept all eleven players separate."
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
excess_deg = {}
for network in ("worm", "email"):
    null = null_values("null_degeneracy", network)
    if len(null):
        deg = value("micro_degeneracy", network=network)
        excess_deg[network] = (deg - null.mean(), beats(deg, null), len(null))
if len(excess_deg) == 2:
    (w_ex, w_b, w_n), (e_ex, e_b, e_n) = excess_deg["worm"], excess_deg["email"]
    bullets.append(
        f"**The worm and email are both more degenerate than chance; only email's redundancy "
        f"can be merged.** Against its rewired versions the worm has {w_ex:+.2f} bits of extra "
        f"degeneracy (above {w_b} of {w_n}) and email {e_ex:+.2f} (above {e_b} of {e_n}). "
        "The worm's overlap is spread thinly across neurons whose targets partly coincide, so "
        "no macro-node captures it; email's sits in large groups of identical addresses."
    )
fb_match = per_match("degeneracy_pct", "ei_pct", "greedy_groups")
if not fb_match.empty:
    n = len(fb_match)
    ei_hi = int((fb_match["ei_pct"] > 0.975).sum())
    deg_hi = int((fb_match["degeneracy_pct"] > 0.975).sum())
    deg_lo = int((fb_match["degeneracy_pct"] < 0.025).sum())
    det, det_null = (
        value("micro_determinism", network="football"),
        null_reference("football", "determinism"),
    )
    roles_loss = -value("grouping_ce", network="football", grouping="roles")
    unmerged = int((fb_match["greedy_groups"] >= value("nodes", network="football")).sum())
    bullets.append(
        f"**Arsenal's passing is specific because of determinism, not low redundancy.** Its "
        f"EI is above the null in {ei_hi} of {n} matches, but its degeneracy is above the "
        f"null in {deg_hi} and below it in {deg_lo}: about what pass volumes produce. The "
        f"gain is in determinism (median {det:.2f} bits against {det_null:.2f} for the "
        "null): each player concentrates passes on particular teammates. With no redundancy "
        f"beyond chance, positional roles lose {roles_loss:.2f} bits in the median match and "
        f"the greedy search merges nobody in {unmerged} of {n}."
    )
if not math.isnan(lv_ce):
    bullets.append(
        f"**Communities are informative, but not macro-nodes in Klein's sense.** Louvain's "
        f"communities lose {-lv_ce:.2f} bits against individual addresses, yet raise "
        f"effectiveness to {lv_eff:.2f}. They are far better than random groupings of the "
        "same sizes, and the rise in effectiveness is the trap shown above."
    )
st.markdown("\n".join(f"- {b}" for b in bullets))

email_wins = bool(len(email_null)) and email_ce > email_null.max()
worm_wins = bool(len(worm_null)) and worm_ce > worm_null.max()
shares = {n: degeneracy_share(n, value("micro_degeneracy", network=n)) for n in NETWORKS}
ordered = sorted(NETWORKS, key=lambda n: shares[n])
st.subheader("Against Klein and Hoel's patterns")
if email_wins and not worm_wins and ordered[-1] == "email":
    ranked = ", ".join(f"{NETWORKS[n][0]} {shares[n]:.2f}" for n in ordered)
    st.write(
        "My three networks fit their second pattern and run against their first. Ranked by "
        f"degeneracy as a share of its maximum ({ranked}), only the most degenerate network "
        "gains from macro-nodes beyond chance, as they predict. But that network is the "
        "email network, not the connectome. I don't read this as a counterexample. Three "
        "networks can't test a trend across domains, Enron is a social network rather than "
        "a technological one, a football team is neither, and this connectome is an "
        "unweighted, 131-neuron subset of the worm. Weighted synapse counts or the whole "
        "nervous system could change the worm's result. What the three do show is the "
        "mechanism: macro-nodes beat micro-nodes when the micro scale is full of redundant "
        "nodes, and the redundancy has to exceed what the degrees alone produce."
    )
else:
    st.write(
        "Three networks can't test a trend across domains, so I compare this run with "
        "Klein and Hoel's patterns only through the tables above: which network is more "
        "degenerate than its null, and which gains from macro-nodes beyond it."
    )

st.subheader("Limitations")
st.markdown(
    "- The worm and email networks are unweighted, and the email network is undirected.\n"
    "- Arsenal's chains are estimated from about 300 passes per match, and entropy-based "
    "measures are biased upward at that size (by about 0.2 bits of EI, as the football "
    "entry shows). Its nulls have the same sample size, so the comparisons with chance are "
    "fair, but the absolute shares are rough next to the worm's and email's fixed wiring.\n"
    "- The greedy search can miss better groupings, and for email I tested two hand-picked "
    "groupings rather than searching. The email gain is a lower bound.\n"
    "- 20 and 10 rewired networks give the null's spread, not precise tail probabilities.\n"
    "- Arsenal's null is a configuration model for counted passes, and the worm's and "
    "email's are degree-preserving rewirings of links. They are analogous, not identical.\n"
    "- Macro-nodes average members' outputs evenly, matching EI's even intervention. "
    "For the worm and email, weighting by how often the walk visits each node gave the "
    "same conclusions when I explored this; for Arsenal, the football entry's "
    "pass-weighted version reaches the same answer (roles and lines lose information)."
)

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
theme = "dark" if st.context.theme.type == "dark" else "light"
st.image(ASSETS / f"statsbomb-logo-{theme}.png", width=220)
st.caption("Arsenal data: StatsBomb open data. Analysis and opinions are my own, not StatsBomb's.")
