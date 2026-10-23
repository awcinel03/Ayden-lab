import json

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lab.gravity_mass import model
from lab.registry import entry
from lab.ui import REPO_URL, page_header

ENTRY = entry("gravity-mass-information-flow")
CODE_PATHS = (
    "lab/gravity_mass/model.py",
    "pipelines/experiments/network_science/gravity_mass.py",
    "tests/test_gravity_mass.py",
)
# Categorical slots 1-3 (validated palette, as on the causal-emergence page); gray for references.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
GRAPH_COLORS = {"BA": BLUE, "ER": ORANGE}
RHO_COLORS = {0.0: BLUE, 0.5: ORANGE, 1.0: AQUA}
GRAPHS = {
    "BA": "Barabási–Albert (m = 3)",
    "ER": "Erdős–Rényi (⟨k⟩ = 6)",
    "mouse": "Mouse connectome backbone",
}
FIELDS = {
    "rho": "ρ dial",
    "heat": "Heat kernel",
    "volume": "Region volume",
    "fa": "Fractional anisotropy",
}


@st.cache_data(max_entries=32)
def network(graph: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Adjacency and a fixed 2-D layout. Synthetic graphs use seed 1, as in the full runs."""
    if graph == "mouse":
        A_full, _, _ = model.load_mouse()
        A = model.knn_backbone(A_full, 6)
    else:
        G = (
            nx.barabasi_albert_graph(n, 3, seed=1)
            if graph == "BA"
            else nx.gnm_random_graph(n, 3 * n, seed=1)
        )
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        A = nx.to_numpy_array(G)
    pos = nx.spring_layout(nx.from_numpy_array((A > 0).astype(float)), seed=0)
    return A, np.array([pos[i] for i in range(len(A))])


@st.cache_data(max_entries=64)
def gravity(graph: str, n: int, field: str, par: float, seed: int) -> np.ndarray:
    """Standardized gravity field g on the chosen network."""
    A, _ = network(graph, n)
    if field == "rho":
        return model.rho_field(A, par, seed)
    if field == "heat":
        B = (A > 0).astype(float)  # unweighted Laplacian, so t means the same on every graph
        evals, evecs = np.linalg.eigh(np.diag(B.sum(1)) - B)
        return model.heat_field(evals, evecs, par, seed)
    _, vol, fa = model.load_mouse()
    return model.standardize(np.log(vol) if field == "volume" else fa)


@st.cache_data(max_entries=64)
def hitting_curve(graph: str, n: int, field: str, par: float, seed: int) -> np.ndarray:
    """Hitting time to T over the full lambda grid (0 to 8 in steps of 0.25)."""
    A, _ = network(graph, n)
    return model.mfpt_curve(A, gravity(graph, n, field, par, seed))


@st.cache_data
def full_runs() -> dict:
    """Precomputed step 3 results; regenerate only with the offline experiment."""
    return json.loads(model.RESULTS_PATH.read_text())


page_header(ENTRY)

st.header("The idea")
st.write(
    "Every node in a network has a *gravity* g, and every message has a *mass* m. A "
    "massless message does a simple random walk; a heavy one is pulled toward "
    "high-gravity neighbours, with the pull set by λ = βm. Gravity is standardized to mean 0 "
    "and variance 1, and the destination T is the top 10% of nodes by g. The question is how "
    "fast messages of different mass reach T, and why there is a best mass: heavy messages "
    "climb quickly but get stuck on local peaks of g outside T (*traps*)."
)
st.latex(
    r"P_{ij}(\lambda) = \frac{A_{ij}\,e^{\lambda g_j}}{\sum_k A_{ik}\,e^{\lambda g_k}},"
    r"\qquad \lambda = \beta m"
)
st.write(
    "On an undirected graph this is a simple random walk with edge conductances "
    "c_ij = A_ij e^{λ(g_i + g_j)}, a gravity law on the graph, so its stationary distribution "
    "has a closed form:"
)
st.latex(r"\pi_i(\lambda) \propto e^{\lambda g_i} \sum_k A_{ik}\, e^{\lambda g_k}")

st.header("Explore the walk")
with st.sidebar:
    st.subheader("Explorer")
    graph = st.selectbox("Graph", list(GRAPHS), format_func=GRAPHS.get, key="gm_graph")
    synthetic = graph != "mouse"
    n = st.slider(
        "Nodes (n)",
        100,
        500,
        300,
        step=50,
        disabled=not synthetic,
        help="Synthetic graphs only. The mouse backbone has 332 regions.",
        key="gm_n",
    )
    field_options = ["rho", "heat"] + ([] if synthetic else ["volume", "fa"])
    field = st.selectbox("Gravity field", field_options, format_func=FIELDS.get, key="gm_field")
    if field == "rho":
        par = st.slider(
            "ρ (correlation with log degree)",
            0.0,
            1.0,
            0.0,
            step=0.1,
            help="g = z(ρ·z(log k) + √(1−ρ²)·ξ), ξ standard normal noise.",
            key="gm_rho",
        )
    elif field == "heat":
        par = st.slider(
            "Heat-kernel time t",
            0.25,
            4.0,
            0.5,
            step=0.25,
            help="g = z(e^{−tL} ξ): larger t gives a smoother field with fewer traps.",
            key="gm_t",
        )
    else:
        par = 0.0
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=999,
        value=0,
        step=1,
        disabled=field in ("volume", "fa"),
        help="Draws the noise ξ in the gravity field. Graphs always use seed 1.",
        key="gm_seed",
    )
    lam = st.slider("Mass λ", 0.0, 8.0, 2.0, step=0.25, key="gm_lam")

n_eff = n if synthetic else 0  # the mouse network ignores n; keep one cache entry for it
A, pos = network(graph, n_eff)
g = gravity(graph, n_eff, field, par, int(seed))
T = model.target_of(g)
h = hitting_curve(graph, n_eff, field, par, int(seed))
i_star = int(np.argmin(h))
lam_star = float(model.LAMS[i_star])
censored = i_star == len(model.LAMS) - 1
P = model.kernel(A, g, lam)
pi = model.stationary_closed_form(A, g, lam)

with st.container(horizontal=True):
    st.metric(
        "Best mass λ*",
        f"≥ {lam_star:g}" if censored else f"{lam_star:g}",
        help="λ with the lowest hitting time on the grid 0–8. At 8 it is a lower bound.",
        border=True,
    )
    st.metric(
        "Trap share",
        f"{model.local_max_fraction(A, g, T):.1%}",
        help="Nodes that are local maxima of g but not in T.",
        border=True,
    )
    st.metric(
        "Entropy rate",
        f"{model.entropy_rate(P, pi):.2f} nats",
        help="Average surprise of one step at the current λ.",
        border=True,
    )
    st.metric(
        "Effective information",
        f"{model.effective_information(P):.2f} bits",
        help="Klein–Hoel EI of P(λ): entropy of the average row minus average row entropy.",
        border=True,
    )

net_col, curve_col = st.columns([3, 2])
with net_col:
    iu, ju = np.nonzero(np.triu(A))
    ex = np.column_stack([pos[iu, 0], pos[ju, 0], np.full(len(iu), np.nan)]).ravel()
    ey = np.column_stack([pos[iu, 1], pos[ju, 1], np.full(len(iu), np.nan)]).ravel()
    size = 4 + 26 * np.sqrt(pi / pi.max())
    fig = go.Figure(
        [
            go.Scattergl(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(color=GRAY, width=0.5),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scattergl(
                x=pos[:, 0],
                y=pos[:, 1],
                mode="markers",
                marker=dict(
                    size=size,
                    color=g,
                    colorscale="RdBu_r",
                    cmid=0,
                    colorbar=dict(title="g", thickness=12),
                    line=dict(color=np.where(T, "#111111", "rgba(0,0,0,0)"), width=2),
                ),
                customdata=np.column_stack([np.arange(len(g)), g, pi, T]),
                hovertemplate=(
                    "node %{customdata[0]:.0f}<br>g = %{customdata[1]:.2f}"
                    "<br>π = %{customdata[2]:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
    )
    st.plotly_chart(fig, key="gm_network", width="stretch", theme="streamlit")
    st.caption(
        f"{len(g)} nodes. Colour is gravity g, size is the stationary probability π(λ) at "
        f"λ = {lam:g}, and the {T.sum()} outlined nodes are the target set T."
    )
with curve_col:
    fig = go.Figure(
        go.Scatter(
            x=model.LAMS,
            y=h,
            mode="lines",
            line=dict(color=BLUE, width=2),
            hovertemplate="λ = %{x}<br>hitting time = %{y:.1f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=lam_star, line=dict(color=AQUA, dash="dash"))
    fig.add_annotation(x=lam_star, y=1, yref="paper", text="λ*", showarrow=False, yshift=8)
    h_now = model.mfpt_to_set(P, T)
    fig.add_trace(
        go.Scatter(
            x=[lam],
            y=[h_now],
            mode="markers",
            marker=dict(color=ORANGE, size=11),
            hovertemplate="current λ = %{x}<br>hitting time = %{y:.1f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="mass λ",
        yaxis=dict(title="mean steps to reach T", type="log"),
    )
    st.plotly_chart(fig, key="gm_curve", width="stretch", theme="streamlit")
    st.caption(
        f"Mean first-passage time to T from a uniform start outside it. Now {h_now:.1f} "
        f"steps at λ = {lam:g}, against {h[0]:.1f} for a massless message."
    )
if graph == "mouse":
    st.caption(
        "Mouse DTI connectome (subject N54776), cut to each region's 6 strongest connections "
        "because the raw graph is 66% dense. Volume and FA come from the subject's label "
        "statistics; the node-to-region mapping behind them is inferred, not documented."
    )

st.header("Results from the full runs")
res = full_runs()
st.write(
    "These charts read the saved output of the offline experiments: BA and ER graphs with "
    "n = 1,000, λ from 0 to 8 in steps of 0.25, and several seeds per setting. Nothing here "
    "is recomputed."
)

st.subheader("Rougher landscapes need lighter messages")
A_rows = pd.DataFrame(res["A"])
fig = go.Figure()
for (gname, kind), sub in A_rows.groupby(["graph", "kind"]):
    base = "circle" if kind == "rho" else "diamond"
    fig.add_trace(
        go.Scatter(
            x=sub["roughness"],
            y=sub["lam_star"],
            mode="markers",
            name=f"{gname}, {'ρ dial' if kind == 'rho' else 'heat kernel'}",
            marker=dict(
                color=GRAPH_COLORS[gname],
                size=8,
                symbol=np.where(sub["censored"], f"{base}-open", base),
                line=dict(width=1.5, color=GRAPH_COLORS[gname]),
            ),
            customdata=sub[["par", "seed"]],
            hovertemplate=(
                "trap share %{x:.1%}<br>λ* = %{y}<br>parameter %{customdata[0]}, "
                "seed %{customdata[1]}<extra></extra>"
            ),
        )
    )
fig.update_layout(
    height=420,
    xaxis=dict(title="trap share (local maxima of g outside T)", tickformat=".0%"),
    yaxis_title="best mass λ*",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, key="gm_roughness", width="stretch", theme="streamlit")
uncensored = A_rows[~A_rows["censored"]]
smooth = A_rows[A_rows["roughness"] < 0.01]
st.caption(
    f"{len(A_rows)} settings: {A_rows.groupby(['graph', 'kind', 'par']).ngroups} fields × 8 "
    f"seeds. Hollow markers ({A_rows['censored'].sum()}) are censored: the minimum sat at "
    f"λ = 8, the edge of the grid, so λ* is at least 8. Below 1% trap share, "
    f"{smooth['censored'].mean():.0%} of settings are censored; uncensored λ* runs from "
    f"{uncensored['lam_star'].max():g} down to {uncensored['lam_star'].min():g}."
)

st.subheader("The barrier sets the cost of being too heavy")
B_rows = pd.DataFrame(res["B"])
B_rows["rel_err"] = (B_rows["slope"] - B_rows["gamma_pred"]).abs() / B_rows["gamma_pred"]
st.write(
    "At large λ the walk behaves like a particle escaping a potential well: the log hitting "
    "time grows linearly in λ, with slope set by the deepest trap (Arrhenius' law). The "
    "predicted barrier Γ* is the largest drop, over nodes outside T, from a node's best edge "
    "to the best path out of it, on the edge landscape g_u + g_v."
)
lo = min(B_rows["gamma_pred"].min(), B_rows["slope"].min()) * 0.9
hi = max(B_rows["gamma_pred"].max(), B_rows["slope"].max()) * 1.05
fig = go.Figure(
    go.Scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        line=dict(color=GRAY, dash="dash"),
        name="y = x",
        hoverinfo="skip",
    )
)
for gname, sub in B_rows.groupby("graph"):
    fig.add_trace(
        go.Scatter(
            x=sub["gamma_pred"],
            y=sub["slope"],
            mode="markers",
            name=gname,
            marker=dict(color=GRAPH_COLORS[gname], size=10),
            customdata=sub[["seed"]],
            hovertemplate=(
                "seed %{customdata[0]}<br>predicted Γ* = %{x:.2f}<br>fitted slope = %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )
fig.update_layout(
    height=400,
    xaxis_title="predicted barrier Γ*",
    yaxis_title="fitted slope of log hitting time (λ = 10–12)",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, key="gm_arrhenius", width="stretch", theme="streamlit")
st.caption(
    f"ρ = 0 fields, 6 seeds per graph. {(B_rows['rel_err'] < 0.07).sum()} of {len(B_rows)} "
    f"fitted slopes are within 7% of the prediction; the median gap is "
    f"{B_rows['rel_err'].median():.1%}."
)

st.subheader("Effective information rises with mass unless gravity tracks degree")
E_rows = pd.DataFrame(res["E"]).groupby(["graph", "rho", "lam"], as_index=False)["ei"].mean()
fig = make_subplots(rows=1, cols=2, subplot_titles=("BA", "ER"), shared_yaxes=True)
for col, gname in enumerate(("BA", "ER"), start=1):
    for rho, color in RHO_COLORS.items():
        sub = E_rows[(E_rows["graph"] == gname) & (E_rows["rho"] == rho)]
        fig.add_trace(
            go.Scatter(
                x=sub["lam"],
                y=sub["ei"],
                mode="lines+markers",
                name=f"ρ = {rho:g}",
                line=dict(color=color),
                legendgroup=str(rho),
                showlegend=col == 1,
                hovertemplate=f"ρ = {rho:g}<br>λ = %{{x}}<br>EI = %{{y:.2f}} bits<extra></extra>",
            ),
            row=1,
            col=col,
        )
fig.update_xaxes(title_text="mass λ")
fig.update_yaxes(title_text="EI (bits)", row=1, col=1)
fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, key="gm_ei", width="stretch", theme="streamlit")
ei_at = E_rows.set_index(["graph", "rho", "lam"])["ei"]
st.caption(
    "Mean over 5 field seeds. With gravity independent of structure (ρ = 0), EI on BA goes "
    f"from {ei_at['BA', 0.0, 0]:.2f} to {ei_at['BA', 0.0, 8]:.2f} bits between λ = 0 and 8; "
    f"with ρ = 1 it moves from {ei_at['BA', 1.0, 0]:.2f} to {ei_at['BA', 1.0, 8]:.2f}, "
    f"bottoming at {E_rows[(E_rows['graph'] == 'BA') & (E_rows['rho'] == 1.0)]['ei'].min():.2f}."
)

st.subheader("Contagion: push spreading on the same kernel")
C_rows = pd.DataFrame(res["C"])
C = C_rows.groupby(["graph", "lam"], as_index=False)[["t_half", "wasted", "walker_mfpt"]].mean()
st.write(
    "Instead of one walker, every informed node pushes the message to one neighbour drawn "
    "from P(λ) each step. Wasted pushes land on nodes that already have the message, counted "
    "until half of T is informed."
)
st.dataframe(
    C,
    hide_index=True,
    column_config={
        "graph": "Graph",
        "lam": st.column_config.NumberColumn("Mass λ", format="%g"),
        "t_half": st.column_config.NumberColumn("Steps to inform 50% of T", format="%.1f"),
        "wasted": st.column_config.ProgressColumn(
            "Wasted pushes", format="percent", min_value=0.0, max_value=1.0
        ),
        "walker_mfpt": st.column_config.NumberColumn("Single-walker hitting time", format="%.1f"),
    },
)
fastest = C.loc[C.groupby("graph")["t_half"].idxmin()].set_index("graph")
st.caption(
    "ρ = 0 fields, 5 seeds × 60 starts per graph and λ. Spreading is fastest at "
    + " and ".join(f"λ = {fastest.loc[gn, 'lam']:g} on {gn}" for gn in fastest.index)
    + f"; wasted pushes rise from {C[C['lam'] == 0]['wasted'].mean():.0%} at λ = 0 to "
    f"{C[C['lam'] == C['lam'].max()]['wasted'].mean():.0%} at λ = {C['lam'].max():g}."
)

st.header("Related work")
st.markdown(
    "- Gómez-Gardeñes, J. & Latora, V. (2008). Entropy rate of diffusion processes on "
    "complex networks. *Physical Review E* 78, 065102(R). The degree-biased walk, the "
    "special case g = log k.\n"
    "- Burda, Z., Duda, J., Luck, J. M. & Waclaw, B. (2009). Localization of the maximal "
    "entropy random walk. *Physical Review Letters* 102, 160602. MERW, the special case "
    "g = log ψ at λ = 1.\n"
    "- Baronchelli, A., Barrat, A. & Pastor-Satorras, R. (2009). Glass transition and random "
    "walks on complex energy landscapes. *Physical Review E* 80, 020102(R). Trap models on "
    "networks; a possible sharp transition in mass, not tested here.\n"
    "- Brockmann, D. & Helbing, D. (2013). The hidden geometry of complex, network-driven "
    "contagion phenomena. *Science* 342, 1337–1342. Effective distance, which predicts "
    "arrival far better than hop count once λ > 0.\n"
    "- Klein, B. & Hoel, E. (2020). The emergence of informative higher scales in complex "
    "networks. *Complexity* 2020, 8932526. Effective information of a network's random walk.\n"
    "- Wang, N. et al. (2020). Variability and heritability of mouse brain structure: "
    "microscopic MRI atlases and connectomes for diverse strains. *NeuroImage* 222, 117274. "
    "[doi:10.1016/j.neuroimage.2020.117274](https://doi.org/10.1016/j.neuroimage.2020.117274). "
    "Source of the mouse connectome."
)

st.header("Code and data")
st.markdown(
    f"The model, data and saved results live in [`lab/gravity_mass/`]"
    f"({REPO_URL}/tree/main/lab/gravity_mass). "
    + " · ".join(f"[{p}]({REPO_URL}/blob/main/{p})" for p in CODE_PATHS)
)
st.caption(
    "Mouse DTI connectome, subject N54776 (C57BL/6J), 332 Waxholm regions, as bundled with "
    "the graspologic 0.3.1 package (MIT licence; no separate licence stated for the data). "
    "The full runs take about 20 minutes and are not rerun by this page."
)
