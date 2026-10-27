import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lab.gravity_mass import model
from lab.registry import entry
from lab.ui import REPO_URL, empty_state, page_header

ENTRY = entry("gravity-mass-follow-ups")
GLASS_PATH = model.RESULTS_PATH.parent / "glass_results.json"
SORTING_PATH = model.RESULTS_PATH.parent / "sorting_results.json"
COARSE_PATH = model.RESULTS_PATH.parent / "coarse_results.json"
TWITCH_PATH = model.RESULTS_PATH.parent / "twitch_results.json"
CODE_PATHS = (
    "lab/gravity_mass/model.py",
    "pipelines/experiments/network_science/gravity_mass_glass.py",
    "pipelines/experiments/network_science/gravity_mass_sorting.py",
    "pipelines/experiments/network_science/gravity_mass_coarse.py",
    "pipelines/experiments/network_science/gravity_mass_twitch.py",
)
# Categorical slots 1-4 (validated palette); network sizes use one sequential blue ramp.
BLUE, ORANGE, AQUA, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8e5cd9", "#8a8a85"
FIELD_COLORS = {"gauss": BLUE, "exp_deep": ORANGE, "exp_shallow": AQUA, "laplace": PURPLE}
SHORT = {
    "gauss": "Normally distributed",
    "exp_deep": "Exponential, deep tail",
    "exp_shallow": "Exponential, shallow tail",
    "laplace": "Laplace",
}
SIZE_RAMP = ["#a9c9f2", "#6fa3e6", "#2a78d6", "#1a4f94"]


@st.cache_data
def twitch() -> dict | None:
    """Raw follow-up D output, or None before the experiment has been run."""
    return json.loads(TWITCH_PATH.read_text()) if TWITCH_PATH.exists() else None


@st.cache_data
def coarse() -> dict | None:
    """Raw follow-up C output, or None before the experiment has been run."""
    return json.loads(COARSE_PATH.read_text()) if COARSE_PATH.exists() else None


@st.cache_data
def coarse_frame() -> pd.DataFrame:
    """One row per setting, seed and mass, with causal emergence and effectiveness gains."""
    res = coarse()
    C = pd.DataFrame(
        [
            dict(
                graph=r["graph"],
                kind=r["kind"],
                par=r["par"],
                seed=r["seed"],
                n=r["n"],
                k=r["k"],
                **x,
            )
            for r in res["runs"]
            for x in r["rows"]
        ]
    )
    C["ce"] = C["macro_ei"] - C["micro_ei"]
    C["advantage"] = C["macro_ei"] - C["null_ei"]
    log_k = np.log2(C["k"].where(C["k"] > 1))
    C["eff_gain"] = C["macro_ei"] / log_k - C["micro_ei"] / np.log2(C["n"])
    C["null_eff_gain"] = C["null_ei"] / log_k - C["micro_ei"] / np.log2(C["n"])
    return C


def coarse_crossings(C: pd.DataFrame) -> dict:
    """How many settings' flow basins (and random groupings) overtake the nodes per state."""
    first, null_crossed, total = [], 0, 0
    for _, sub in C[C["k"] > 1].groupby(["graph", "kind", "par", "seed"]):
        total += 1
        up = sub[sub["eff_gain"] > 0]
        if len(up):
            first.append(up["lam"].min())
        null_crossed += int((sub["null_eff_gain"] > 0).any())
    return dict(
        crossed=len(first),
        total=total,
        median=float(np.median(first)) if first else float("nan"),
        null_crossed=null_crossed,
    )


@st.cache_data
def sorting() -> dict | None:
    """Raw follow-up B output, or None before the experiment has been run."""
    return json.loads(SORTING_PATH.read_text()) if SORTING_PATH.exists() else None


@st.cache_data
def glass() -> dict | None:
    """Raw follow-up A output, or None before the experiment has been run."""
    return json.loads(GLASS_PATH.read_text()) if GLASS_PATH.exists() else None


@st.cache_data
def glass_summary(n_boot: int = 300) -> dict:
    """Means, finite-size exponents with bootstrap bands, and tail fits, per field."""
    res = glass()
    lams = np.array(res["config"]["lams"])
    runs = res["runs"]
    rng = np.random.default_rng(0)
    out = {}
    for kind in res["config"]["fields"]:
        by_n = {}
        for r in runs:
            if r["field"] == kind:
                by_n.setdefault(r["n_nominal"], []).append([row["y2_basin"] for row in r["rows"]])
        ns = np.array(sorted(by_n))
        Y = {n: np.array(by_n[n]) for n in ns}
        mean = np.array([Y[n].mean(0) for n in ns])

        def slopes(means, ns=ns):
            return np.polyfit(np.log(ns), np.log(means), 1)[0]

        expo = slopes(mean)
        boot = np.array(
            [
                slopes(np.array([Y[n][rng.integers(0, len(Y[n]), len(Y[n]))].mean(0) for n in ns]))
                for _ in range(n_boot)
            ]
        )
        climbs = np.concatenate([r["climbs"] for r in runs if r["field"] == kind])
        u = np.median(climbs)
        gamma0 = float((climbs[climbs > u] - u).mean())  # exponential-tail scale above median
        cross = np.flatnonzero((expo[:-1] < -0.5) & (expo[1:] >= -0.5))
        lam_fs = (
            float(np.interp(-0.5, expo[cross[0] : cross[0] + 2], lams[cross[0] : cross[0] + 2]))
            if len(cross)
            else None
        )
        out[kind] = dict(
            ns=ns.tolist(),
            mean=mean.tolist(),
            expo=expo.tolist(),
            lo=np.quantile(boot, 0.05, axis=0).tolist(),
            hi=np.quantile(boot, 0.95, axis=0).tolist(),
            climbs=np.sort(climbs).tolist(),
            gamma0=gamma0,
            lam_pred=1 / gamma0,
            lam_fs=lam_fs,
        )
    return {"lams": lams.tolist(), "fields": out}


page_header(ENTRY)
st.write(
    "Four questions left open by the first entry, each answered with the same model: nodes "
    "are gravity wells of depth g, a message of mass λ steps to neighbour j with probability "
    "∝ A_ij e^{λ g_j}, and the destination T is the deepest 10% of wells."
)
st.page_link(
    entry("gravity-mass-information-flow").path,
    label="Back to the first entry: Gravity and Mass",
    icon=":material/arrow_back:",
)

tab_glass, tab_sort, tab_social, tab_coarse = st.tabs(
    ["Glass transition", "Mass sorting", "A social network", "Coarse-graining"]
)

with tab_glass:
    st.header("Is there a glass-like transition in mass?")
    res = glass()
    if res is None:
        empty_state("No results yet. Run the glass-transition experiment to fill this section.")
    else:
        S = glass_summary()
        lams = np.array(S["lams"])
        F = S["fields"]
        cfg = res["config"]

        st.subheader("The question")
        st.write(
            "The first entry found that heavy messages slow down exponentially, at a rate set "
            "by the single deepest trap. That is a smooth slowdown. The trap model of glassy "
            "physics predicts something sharper. Escaping a well with climb Γ takes about "
            "e^{λΓ} steps. If climbs have an exponential tail, P(Γ > x) ≈ e^{−x/Γ₀}, the "
            "average escape time over all wells is finite only while λ < 1/Γ₀. Past that "
            "critical mass λ_c = 1/Γ₀ the time before arrival *condenses*: it is spent almost "
            "entirely in a handful of the deepest wells, however large the network."
        )
        st.latex(
            r"\lambda_c = 1/\Gamma_0, \qquad"
            r"\langle e^{\lambda \Gamma} \rangle < \infty \iff \lambda < \lambda_c"
        )

        st.subheader("How I tested it")
        st.markdown(
            f"- **Networks:** Erdős–Rényi graphs with mean degree 6 at n = "
            f"{', '.join(f'{n:,}' for n in cfg['sizes'])}, with "
            f"{cfg['seeds']['500']} seeds per size up to 2,000 and {cfg['seeds']['4000']} at "
            "4,000. These sizes need the new sparse solver; it matches the dense one to 1e-9.\n"
            "- **Four distributions of well depth**, all standardized: normally distributed, "
            "exponential with its long tail toward deep wells, exponential with its long tail "
            "toward shallow ground, and Laplace (both tails exponential).\n"
            "- **Climbs:** for every trap (a local well outside T), the floor-to-rim climb from "
            "the first entry's Arrhenius test.\n"
            "- **Order parameter:** group nodes into *basins*, each node joining the well it "
            "drains into by always stepping to its deepest neighbour. With wᵦ the share of "
            "pre-arrival time spent in basin b (averaged over uniform starts), the "
            "participation ratio is Y₂ = Σ wᵦ². If time is spread over many basins, Y₂ ≈ "
            "1/(number of basins) and shrinks as n grows. If it condenses, Y₂ stays finite.\n"
            "- **Finite-size test:** the slope of log Y₂ against log n at each mass. About −1 "
            "means spread out; about 0 means condensed. A true transition is a jump between the "
            "two at λ_c that sharpens as n grows."
        )

        st.subheader("Climbs have exponential-looking tails")
        fig = go.Figure()
        for kind, f in F.items():
            c = np.array(f["climbs"])
            surv = 1 - np.arange(len(c)) / len(c)
            fig.add_trace(
                go.Scatter(
                    x=c,
                    y=surv,
                    mode="lines",
                    name=SHORT[kind],
                    line=dict(color=FIELD_COLORS[kind], width=2),
                    hovertemplate=f"{SHORT[kind]}<br>Γ = %{{x:.2f}}<br>P(climb > Γ) = %{{y:.3f}}"
                    "<extra></extra>",
                )
            )
        fig.update_layout(
            height=380,
            xaxis_title="climb Γ (floor to rim)",
            yaxis=dict(title="share of traps with a larger climb", type="log"),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, key="glass_tail", width="stretch", theme="streamlit")
        st.caption(
            "On a log scale an exponential tail is a straight line, with slope −1/Γ₀. "
            + "; ".join(
                f"{SHORT[k]}: {len(f['climbs']):,} traps, Γ₀ = {f['gamma0']:.2f}"
                for k, f in F.items()
            )
            + ". Γ₀ is the mean excess above the median climb."
        )

        st.subheader("Time condenses into a few wells as mass grows")
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[SHORT[k] for k in F],
            shared_xaxes=True,
            shared_yaxes=True,
        )
        for idx, f in enumerate(F.values()):
            r, c = divmod(idx, 2)
            for j, (n, y) in enumerate(zip(f["ns"], f["mean"], strict=True)):
                fig.add_trace(
                    go.Scatter(
                        x=lams,
                        y=y,
                        mode="lines",
                        name=f"n = {n:,}",
                        legendgroup=str(n),
                        showlegend=idx == 0,
                        line=dict(color=SIZE_RAMP[j], width=2),
                        hovertemplate=f"n = {n:,}<br>λ = %{{x}}<br>Y₂ = %{{y:.3f}}<extra></extra>",
                    ),
                    row=r + 1,
                    col=c + 1,
                )
            pred = np.where(lams > f["lam_pred"], 1 - f["lam_pred"] / np.maximum(lams, 1e-9), 0)
            fig.add_trace(
                go.Scatter(
                    x=lams,
                    y=pred,
                    mode="lines",
                    name="trap model, n → ∞",
                    legendgroup="pred",
                    showlegend=idx == 0,
                    line=dict(color=GRAY, dash="dash"),
                    hovertemplate="trap model<br>λ = %{x}<br>Y₂ = %{y:.3f}<extra></extra>",
                ),
                row=r + 1,
                col=c + 1,
            )
        fig.update_xaxes(title_text="mass λ", row=2)
        fig.update_yaxes(title_text="Y₂ (basins)", col=1)
        fig.update_layout(height=560, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, key="glass_y2", width="stretch", theme="streamlit")
        st.caption(
            "Mean Y₂ over seeds. Below the crossover the curves fall as n grows (time is spread "
            "over more basins); above it they collapse onto each other (time sits in a few "
            "wells whatever the size). The dashed line is the trap model's infinite-size "
            "prediction, 1 − λ_c/λ, using λ_c = 1/Γ₀ from the climbs."
        )

        st.subheader("The finite-size test")
        fig = go.Figure()
        for kind, f in F.items():
            col = FIELD_COLORS[kind]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([lams, lams[::-1]]),
                    y=np.concatenate([f["hi"], f["lo"][::-1]]),
                    fill="toself",
                    fillcolor=col,
                    opacity=0.15,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=kind,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=lams,
                    y=f["expo"],
                    mode="lines",
                    name=SHORT[kind],
                    legendgroup=kind,
                    line=dict(color=col, width=2),
                    hovertemplate=f"{SHORT[kind]}<br>λ = %{{x}}<br>slope = %{{y:.2f}}"
                    "<extra></extra>",
                )
            )
        for y, label in ((-1, "spread over ~n basins"), (0, "condensed")):
            fig.add_hline(y=y, line=dict(color=GRAY, dash="dot"), annotation_text=label)
        fig.update_layout(
            height=400,
            xaxis_title="mass λ",
            yaxis_title="d log Y₂ / d log n",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, key="glass_expo", width="stretch", theme="streamlit")
        st.caption(
            "Slope of log Y₂ against log n over the four sizes, with a 90% bootstrap band "
            "(resampling seeds). The finite-size estimate of λ_c is where the slope crosses −0.5."
        )

        rows = []
        for kind, f in F.items():
            i10 = len(lams) - 1
            rows.append(
                dict(
                    field=SHORT[kind],
                    gamma0=f["gamma0"],
                    lam_pred=f["lam_pred"],
                    lam_fs=f["lam_fs"],
                    y2_big=f["mean"][-1][i10],
                    y2_pred=max(0.0, 1 - f["lam_pred"] / lams[i10]),
                )
            )
        T = pd.DataFrame(rows)
        st.dataframe(
            T,
            hide_index=True,
            column_config={
                "field": "Depth distribution",
                "gamma0": st.column_config.NumberColumn("Tail scale Γ₀", format="%.2f"),
                "lam_pred": st.column_config.NumberColumn("λ_c = 1/Γ₀", format="%.1f"),
                "lam_fs": st.column_config.NumberColumn("λ_c from finite size", format="%.1f"),
                "y2_big": st.column_config.NumberColumn(
                    f"Y₂ at λ = {lams[-1]:g}, n = {cfg['sizes'][-1]:,}", format="%.2f"
                ),
                "y2_pred": st.column_config.NumberColumn("Trap model Y₂", format="%.2f"),
            },
        )

        st.subheader("What it means")
        fs = {k: f["lam_fs"] for k, f in F.items() if f["lam_fs"] is not None}

        def slope_at(kind, lam):
            return float(np.interp(lam, lams, F[kind]["expo"]))

        st.markdown(
            "- **Condensation happens, and it needs no special landscape.** In all four "
            "distributions the finite-size slope moves from about −1 to about 0 over a narrow "
            "window of mass: "
            "finite-size estimates of λ_c run from "
            f"{min(fs.values()):.1f} to {max(fs.values()):.1f}. With normally distributed depths "
            "the slope is "
            f"{slope_at('gauss', 2):.2f} at λ = 2 and {slope_at('gauss', 6):.2f} at λ = 6. Above "
            "the window, the share of time held by the deepest few wells no longer shrinks with "
            "network size. That is the signature of a glass-like, condensed phase.\n"
            "- **Exponential climbs come for free.** Even with normally distributed depths, trap "
            "climbs have a roughly exponential tail. A climb is a difference between edge depths "
            "(floor minus rim), and extremes of such differences tend toward exponential tails. "
            "So the trap model applies without building it in.\n"
            "- **With normally distributed depths the trap model gets the critical mass right:** "
            "λ_c = 1/Γ₀ = "
            f"{F['gauss']['lam_pred']:.1f} against {F['gauss']['lam_fs']:.1f} from finite-size "
            "scaling.\n"
            "- **For the other distributions it does not.** Their climb tails are not cleanly "
            "exponential: the mean excess drifts with the threshold (down for the deep and "
            "Laplace distributions, up for the shallow one). A single Γ₀ then misses: "
            + "; ".join(
                f"{SHORT[k]} {F[k]['lam_pred']:.1f} predicted, {F[k]['lam_fs']:.1f} measured"
                for k in ("exp_deep", "exp_shallow", "laplace")
            )
            + ". Measuring Γ₀ only on the deepest traps narrows the gap but does not close it "
            "for every distribution at once, so I report the estimator fixed before the full run.\n"
            "- **Condensation is weaker than the trap model expects.** At λ = 10 and n = 4,000, "
            f"Y₂ is {F['gauss']['mean'][-1][-1]:.2f} with normally distributed depths, against "
            f"{1 - F['gauss']['lam_pred'] / lams[-1]:.2f} predicted. Time stays split across a few "
            "wells rather than one.\n"
            "- **Limits:** sizes span a factor of 8 (500 to 4,000 nodes). That shows a clear "
            "crossover but cannot prove a true transition in the infinite-size limit. The "
            "networks are Erdős–Rényi only."
        )

with tab_sort:
    st.header("Do messages sort by mass?")
    srt = sorting()
    if srt is None:
        empty_state("No results yet. Run the mass-sorting experiment to fill this section.")
    else:
        masses = srt["config"]["masses"]
        R = pd.DataFrame(srt["runs"])
        PM = pd.DataFrame(
            [
                dict(graph=r.graph, kind=r.kind, par=r.par, seed=r.seed, **x)
                for r in R.itertuples()
                for x in r.per_mass
            ]
        )
        mouse = {m["field"]: m for m in srt["mouse"]}

        st.subheader("The question")
        st.write(
            "Release messages of many masses from the same node. Do light and heavy ones end "
            "up in different deep wells? In the solar-system picture: does the heaviness of "
            "what you throw decide which gravity well catches it?"
        )

        st.subheader("How I measured it")
        st.markdown(
            "- **Deep basins:** the destination T (deepest 10% of wells) usually splits into "
            "several separate clusters of linked target nodes. Each cluster is a *deep basin*.\n"
            "- **Exact destinations:** for every start and mass, absorption probabilities give "
            "the chance of reaching each deep basin first. These are linear solves, not "
            "simulations, checked against Monte Carlo in the tests.\n"
            f"- **Masses:** {', '.join(f'{m:g}' for m in masses)}, released in equal numbers.\n"
            "- **Destination sorting:** how much a message's mass tells you about the basin it "
            "reaches, given where it started: the mutual information between mass and basin, "
            "divided by the entropy of mass. 0 means mass is irrelevant; 1 means mass alone "
            "decides the basin.\n"
            "- **Settling sorting:** the same quantity for where messages spend their time in "
            "the long run, from the closed-form stationary distribution.\n"
            f"- **Settings:** BA and ER graphs (n = 1,000), "
            f"{R.groupby(['kind', 'par']).ngroups} fields per graph (the ρ dial and the heat "
            f"kernel) with {srt['config']['seeds']} seeds each, plus the mouse connectome."
        )

        st.subheader("Each step up in mass sends messages somewhere new")
        rough = R[(R["kind"] == "rho") & (R["par"] == 0.0)]
        tv = np.mean(np.array(rough["tv"].tolist()), axis=0)
        labels = [f"λ = {m:g}" for m in masses]
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(
                go.Heatmap(
                    z=tv,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    text=np.round(tv, 2),
                    texttemplate="%{text}",
                    colorbar=dict(title="share"),
                    hovertemplate="%{y} vs %{x}<br>destinations differ for %{z:.0%}<extra></extra>",
                )
            )
            fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, key="sort_tv", width="stretch", theme="streamlit")
        with col_b:
            fig = go.Figure()
            for gname, dash in (("BA", "solid"), ("ER", "dash")):
                for par, color in ((0.0, BLUE), (0.6, ORANGE), (1.0, AQUA)):
                    sub = R[(R["graph"] == gname) & (R["kind"] == "rho") & (R["par"] == par)]
                    fig.add_trace(
                        go.Scatter(
                            x=srt["config"]["lams"],
                            y=np.mean(np.array(sub["tv_curve"].tolist()), axis=0),
                            mode="lines",
                            name=f"{gname}, ρ = {par:g}",
                            line=dict(color=color, dash=dash, width=2),
                            hovertemplate=f"{gname}, ρ = {par:g}<br>λ = %{{x}}<br>"
                            "differs from massless: %{y:.0%}<extra></extra>",
                        )
                    )
            fig.update_layout(
                height=380,
                xaxis_title="mass λ",
                yaxis=dict(title="destinations differing from massless", tickformat=".0%"),
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig, key="sort_curve", width="stretch", theme="streamlit")
        steps = [tv[i, i + 1] for i in range(len(masses) - 1)]
        st.caption(
            "Left: total-variation distance between the destination basins of two masses "
            "released from the same start, averaged over starts, on rough fields (ρ = 0, BA and "
            f"ER, 8 seeds each). A massless and a λ = {masses[-1]:g} message reach different deep "
            f"basins {tv[0, -1]:.0%} of the time. Each step up the mass ladder moves "
            f"{min(steps):.0%}–{max(steps):.0%} of destinations. Right: the same distance "
            "against a massless message over the full mass range; on smoother fields there is "
            "less to sort into."
        )

        st.subheader("Sorting needs separate deep basins")
        fig = go.Figure()
        for (gname, kind), sub in R.groupby(["graph", "kind"]):
            fig.add_trace(
                go.Scatter(
                    x=sub["n_basins"],
                    y=sub["dest_sorting"],
                    mode="markers",
                    name=f"{gname}, {'ρ dial' if kind == 'rho' else 'heat kernel'}",
                    marker=dict(
                        color=BLUE if gname == "BA" else ORANGE,
                        symbol="circle" if kind == "rho" else "diamond",
                        size=7,
                        opacity=0.8,
                    ),
                    customdata=sub[["par", "trap_share"]],
                    hovertemplate=(
                        "%{x} deep basins<br>sorting = %{y:.3f}<br>parameter %{customdata[0]}, "
                        "trap share %{customdata[1]:.1%}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            height=400,
            xaxis=dict(title="number of separate deep basins", type="log"),
            yaxis_title="destination sorting (share of mass entropy)",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, key="sort_basins", width="stretch", theme="streamlit")
        rho_bn = R[["n_basins", "dest_sorting"]].corr(method="spearman").iloc[0, 1]
        rho_tr = R[["trap_share", "dest_sorting"]].corr(method="spearman").iloc[0, 1]
        single = R[R["n_basins"] == 1]
        st.caption(
            f"{len(R)} settings. Sorting follows the number of separate deep basins (rank "
            f"correlation {rho_bn:.2f}) more closely than roughness (trap share, {rho_tr:.2f}). "
            f"When the deepest wells form one connected cluster ({len(single)} settings, all "
            "BA with gravity closely tracking degree, ρ ≥ 0.8), there is nothing to sort into "
            "and sorting is 0. "
            f"The largest value is {R['dest_sorting'].max():.2f}: mass shapes destinations, but "
            "the start still matters more."
        )

        st.subheader("Heavy messages take the nearest way in, not the biggest well")
        cases = {
            "BA, ρ = 0": PM[(PM["graph"] == "BA") & (PM["kind"] == "rho") & (PM["par"] == 0.0)],
            "ER, ρ = 0": PM[(PM["graph"] == "ER") & (PM["kind"] == "rho") & (PM["par"] == 0.0)],
            "Mouse, FA": pd.DataFrame(mouse["FA"]["per_mass"]),
        }
        colors = {"BA, ρ = 0": BLUE, "ER, ρ = 0": ORANGE, "Mouse, FA": AQUA}
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Hops to the destination",
                "Depth of the destination node",
                "Reaching the deepest basin",
            ),
        )
        for name, sub in cases.items():
            agg = sub.groupby("lam")[["dest_hops", "dest_depth", "deepest_basin"]].mean()
            for col, metric in enumerate(("dest_hops", "dest_depth", "deepest_basin"), start=1):
                fig.add_trace(
                    go.Scatter(
                        x=agg.index,
                        y=agg[metric],
                        mode="lines+markers",
                        name=name,
                        legendgroup=name,
                        showlegend=col == 1,
                        line=dict(color=colors[name], width=2),
                        hovertemplate=f"{name}<br>λ = %{{x}}<br>%{{y:.3f}}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
        fig.update_xaxes(title_text="mass λ")
        fig.update_yaxes(tickformat=".0%", row=1, col=3)
        fig.update_layout(height=380, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, key="sort_where", width="stretch", theme="streamlit")

        def at(name, lam, metric):
            sub = cases[name]
            return float(sub[sub["lam"] == lam][metric].mean())

        st.caption(
            f"From λ = 0 to {masses[-1]:g}, the trip into T shortens (BA: "
            f"{at('BA, ρ = 0', 0.0, 'dest_hops'):.1f} to "
            f"{at('BA, ρ = 0', masses[-1], 'dest_hops'):.1f} hops) and lands on a deeper target "
            "node, because heavy messages enter T through its deepest nearby edge. But they are "
            "barely more likely to reach the single deepest basin (BA: "
            f"{at('BA, ρ = 0', 0.0, 'deepest_basin'):.1%} to "
            f"{at('BA, ρ = 0', masses[-1], 'deepest_basin'):.1%}), and on the mouse brain with FA "
            f"less likely ({at('Mouse, FA', 0.0, 'deepest_basin'):.0%} to "
            f"{at('Mouse, FA', masses[-1], 'deepest_basin'):.0%}). Light messages wander far "
            "enough to find distant basins; heavy ones fall into the first deep basin downhill."
        )

        st.subheader("Where they settle")
        st.write(
            "Over the long run, position sorts by mass everywhere: settling sorting is "
            f"{R['settle_sorting'].min():.2f}–{R['settle_sorting'].max():.2f} across all "
            "settings, even where destination sorting is 0. Heavy messages pile into the "
            "deepest wells and stay; light ones spread out. So mass always separates messages "
            "eventually, but *which* deep basin they reach first depends on mass only when "
            "there are several to choose from."
        )

        st.subheader("The mouse brain")
        st.dataframe(
            pd.DataFrame(
                [
                    dict(
                        field={
                            "log volume": "Region volume (log)",
                            "FA": "Fractional anisotropy",
                        }.get(k, "Random (control)"),
                        basins=m["n_basins"],
                        trap=m["trap_share"],
                        dest=m["dest_sorting"],
                        settle=m["settle_sorting"],
                    )
                    for k, m in mouse.items()
                ]
            ),
            hide_index=True,
            column_config={
                "field": "Gravity",
                "basins": st.column_config.NumberColumn("Deep basins", format="%d"),
                "trap": st.column_config.NumberColumn("Trap share", format="percent"),
                "dest": st.column_config.NumberColumn("Destination sorting", format="%.3f"),
                "settle": st.column_config.NumberColumn("Settling sorting", format="%.3f"),
            },
        )
        st.caption(
            "Same rule on the 332-region backbone. Region volume tracks the wiring, so its deep "
            "wells form a single cluster and destinations do not sort. FA's deep wells are "
            "scattered over 12 basins and they do. The node-to-region mapping behind volume and "
            "FA is inferred, not documented."
        )

        st.subheader("What it means")
        st.markdown(
            "- **Yes, messages sort by mass**, and smoothly: each step up the mass ladder moves a "
            "similar share of destinations, with no threshold.\n"
            "- **The sorting rule is local.** Heavy messages fall into the nearest deep basin "
            "downhill from where they start; light messages diffuse and reach more distant "
            "ones. Mass does not steer messages toward the single biggest well.\n"
            "- **Sorting needs a fragmented destination.** It tracks the number of separate "
            "deep basins, which is why gravity that follows the hubs (one cluster of deep "
            "wells) cannot sort at all.\n"
            "- **Limits:** the destination is fixed as the top 10% of g, so the number of basins "
            "and the landscape's roughness change together and cannot be fully separated here."
        )
with tab_social:
    st.header("A real social network: Twitch")
    tw = twitch()
    if tw is None:
        empty_state("No results yet. Run the Twitch experiment to fill this section.")
    else:
        net, src = tw["network"], tw["source"]
        tl = np.array(tw["config"]["lams"])
        fields = tw["fields"]
        real = {f["field"]: f for f in fields if f["field"] != "shuffled log views"}
        shuf = [f for f in fields if f["field"] == "shuffled log views"]
        views, age, deg = real["log views"], real["log account age"], real["log degree"]

        def g0(f):
            c = np.array(f["climbs"])
            if len(c) < 10:
                return float("nan")
            u = np.median(c)
            return float((c[c > u] - u).mean())

        def shuf_range(key):
            vals = [f[key] for f in shuf]
            return min(vals), max(vals)

        st.subheader("The network")
        st.write(
            f"Twitch's English-language streamers in May 2018: {net['n']:,} streamers joined "
            f"by {net['edges']:,} mutual friendships (mean {net['mean_degree']:.1f} friends "
            "each). The data was collected from Twitch's public API and published by "
            "Rozemberczki, Allen and Sarkar. Streamers are identified only by numeric IDs, and "
            "this page shows aggregates only."
        )
        st.caption(
            f"Source: [{src['source']}]({src['source_url']}). Licence: {src['license']}. "
            f"Cite: {src['citation']}"
        )

        st.subheader("Popularity as gravity")
        st.markdown(
            "- **log views:** a streamer's total views. Its correlation with log friend count "
            f"is {net['corr_log_views_log_degree']:.2f}, so it sits in the middle of the ρ "
            "dial.\n"
            f"- **log account age:** correlation {net['corr_log_age_log_degree']:.2f} with log "
            "friend count, near the bottom of the dial.\n"
            "- **log degree:** friend count itself, the top of the dial (ρ = 1).\n"
            f"- **Control: shuffled log views.** The same view counts dealt out to random "
            f"streamers ({len(shuf)} shuffles). Everything about the values stays; only *who* "
            "has them changes, so any difference from real views comes from where the popular "
            "streamers sit in the friendship network."
        )
        rows = []
        for name, f in [("log views", views), ("log account age", age), ("log degree", deg)]:
            rows.append(
                dict(
                    field=name,
                    trap=f["trap_share"],
                    basins=f["n_basins"],
                    lam_star=f"≥ {f['lam_star']:g}" if f["censored"] else f"{f['lam_star']:g}",
                    h0=f["curve"][0]["h_mean"],
                    hmin=min(c["h_mean"] for c in f["curve"]),
                )
            )
        lo_t, hi_t = shuf_range("trap_share")
        lo_b, hi_b = shuf_range("n_basins")
        rows.append(
            dict(
                field="shuffled log views",
                trap=float(np.mean([f["trap_share"] for f in shuf])),
                basins=int(np.mean([f["n_basins"] for f in shuf])),
                lam_star=", ".join(sorted({f"{f['lam_star']:g}" for f in shuf})),
                h0=float(np.mean([f["curve"][0]["h_mean"] for f in shuf])),
                hmin=float(np.mean([min(c["h_mean"] for c in f["curve"]) for f in shuf])),
            )
        )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "field": "Gravity",
                "trap": st.column_config.NumberColumn("Trap share", format="percent"),
                "basins": st.column_config.NumberColumn("Deep basins", format="%d"),
                "lam_star": "Best mass λ*",
                "h0": st.column_config.NumberColumn("Steps to T, massless", format="%.1f"),
                "hmin": st.column_config.NumberColumn("Steps to T, best mass", format="%.1f"),
            },
        )
        st.caption(
            f"Real view counts make a much smoother landscape than the same counts shuffled: "
            f"{views['trap_share']:.1%} of streamers are traps against {lo_t:.1%}–{hi_t:.1%}, and "
            f"the top 10% form {views['n_basins']} separate deep basins against {lo_b}–{hi_b}. "
            "Popular streamers are friends with each other, so the deepest wells join into "
            "one large core. Shuffled rows are means over shuffles; best masses list each "
            "shuffle's value."
        )

        st.subheader("The best mass")
        fig = go.Figure()
        for f in shuf:
            fig.add_trace(
                go.Scatter(
                    x=tl,
                    y=[c["h_mean"] for c in f["curve"]],
                    mode="lines",
                    line=dict(color=GRAY, width=1),
                    opacity=0.6,
                    name="shuffled views",
                    legendgroup="shuf",
                    showlegend=f is shuf[0],
                    hovertemplate="shuffled<br>λ = %{x}<br>steps = %{y:.1f}<extra></extra>",
                )
            )
        for name, f, color in (
            ("log views", views, BLUE),
            ("log account age", age, ORANGE),
            ("log degree", deg, AQUA),
        ):
            fig.add_trace(
                go.Scatter(
                    x=tl,
                    y=[c["h_mean"] for c in f["curve"]],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"{name}<br>λ = %{{x}}<br>steps = %{{y:.1f}}<extra></extra>",
                )
            )
        fig.update_layout(
            height=400,
            xaxis_title="mass λ",
            yaxis=dict(title="mean steps to reach T", type="log"),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, key="tw_curve", width="stretch", theme="streamlit")
        st.caption(
            f"Views and account age give the same U shape as the synthetic graphs, with best "
            f"masses of {views['lam_star']:g} and {age['lam_star']:g}, close to the shuffled "
            "controls. Degree has no traps, so heavier keeps helping. Past the best mass, "
            "hitting times explode: at λ = 8 a message takes "
            f"{views['curve'][-1]['h_mean']:,.0f} steps on real views."
        )

        st.subheader("Condensation, sharper than on synthetic graphs")
        fig = go.Figure()
        for f in shuf:
            fig.add_trace(
                go.Scatter(
                    x=tl,
                    y=[c["y2"] for c in f["curve"]],
                    mode="lines",
                    line=dict(color=GRAY, width=1),
                    opacity=0.6,
                    name="shuffled views",
                    legendgroup="shuf",
                    showlegend=f is shuf[0],
                    hovertemplate="shuffled<br>λ = %{x}<br>Y₂ = %{y:.3f}<extra></extra>",
                )
            )
        for name, f, color in (("log views", views, BLUE), ("log account age", age, ORANGE)):
            fig.add_trace(
                go.Scatter(
                    x=tl,
                    y=[c["y2"] for c in f["curve"]],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"{name}<br>λ = %{{x}}<br>Y₂ = %{{y:.3f}}<extra></extra>",
                )
            )
            fig.add_vline(x=1 / g0(f), line=dict(color=color, dash="dot"))
        fig.update_layout(
            height=400,
            xaxis_title="mass λ",
            yaxis_title="Y₂ (time held by the dominant basins)",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, key="tw_y2", width="stretch", theme="streamlit")

        def y2_at(f, lam):
            return float(np.interp(lam, tl, [c["y2"] for c in f["curve"]]))

        st.caption(
            "Participation ratio of pre-arrival time over flow basins, as in the glass-transition "
            "tab. Dotted lines mark the trap model's λ_c = 1/Γ₀ from each field's climbs: "
            f"{1 / g0(views):.1f} for views and {1 / g0(age):.1f} for account age. On real views, "
            f"Y₂ goes from {y2_at(views, 2):.2f} at λ = 2 to {y2_at(views, 6):.2f} at λ = 6: "
            "nearly all the time before arrival sits in one basin. Shuffled views condense too, "
            f"but how far varies a lot between shuffles ({min(y2_at(f, 8) for f in shuf):.2f}–"
            f"{max(y2_at(f, 8) for f in shuf):.2f} at λ = 8), as expected when a few rare deep "
            "traps dominate. With one network there is no finite-size test here."
        )

        st.subheader("No sorting: every road leads to the core")
        srows = []
        for name, f in [("log views", views), ("log account age", age), ("log degree", deg)]:
            srows.append(
                dict(
                    field=name,
                    basins=f["n_basins"],
                    dest=f["dest_sorting"],
                    tv=f["tv_0_8"],
                    deep0=f["per_mass"][0]["deepest_basin"],
                    deep8=f["per_mass"][-1]["deepest_basin"],
                    settle=f["settle_sorting"],
                )
            )
        srows.append(
            dict(
                field="shuffled log views",
                basins=int(np.mean([f["n_basins"] for f in shuf])),
                dest=float(np.mean([f["dest_sorting"] for f in shuf])),
                tv=float(np.mean([f["tv_0_8"] for f in shuf])),
                deep0=float(np.mean([f["per_mass"][0]["deepest_basin"] for f in shuf])),
                deep8=float(np.mean([f["per_mass"][-1]["deepest_basin"] for f in shuf])),
                settle=float(np.mean([f["settle_sorting"] for f in shuf])),
            )
        )
        st.dataframe(
            pd.DataFrame(srows),
            hide_index=True,
            column_config={
                "field": "Gravity",
                "basins": st.column_config.NumberColumn("Deep basins", format="%d"),
                "dest": st.column_config.NumberColumn("Destination sorting", format="%.3f"),
                "tv": st.column_config.NumberColumn(
                    "Light vs heavy: different basin", format="percent"
                ),
                "deep0": st.column_config.NumberColumn(
                    "Reach deepest basin, λ = 0", format="percent"
                ),
                "deep8": st.column_config.NumberColumn(
                    "Reach deepest basin, λ = 8", format="percent"
                ),
                "settle": st.column_config.NumberColumn("Settling sorting", format="%.3f"),
            },
        )
        st.caption(
            f"On real views, {views['per_mass'][0]['deepest_basin']:.0%} of massless and "
            f"{views['per_mass'][-1]['deepest_basin']:.0%} of λ = 8 messages first reach the "
            "basin holding the most-viewed streamer, so a message's mass hardly changes where "
            f"it goes (sorting {views['dest_sorting']:.3f}, against "
            f"{shuf_range('dest_sorting')[0]:.2f}–{shuf_range('dest_sorting')[1]:.2f} for "
            "shuffled views). This matches the sorting tab's rule: sorting needs several "
            "separate deep basins, and a well-connected elite leaves one. Where messages "
            "settle in the long run still sorts by mass. For shuffled views, whether the "
            "most-viewed streamer lands in the big cluster varies between shuffles, so that "
            "mean is noisy."
        )

        st.subheader("Coarse-graining")
        fig = go.Figure()
        for f in shuf:
            k = f["n_flow_basins"]
            fig.add_trace(
                go.Scatter(
                    x=[c["lam"] for c in f["coarse"]],
                    y=[
                        c["macro_ei"] / np.log2(k) - c["micro_ei"] / np.log2(net["n"])
                        for c in f["coarse"]
                    ],
                    mode="lines",
                    line=dict(color=GRAY, width=1),
                    opacity=0.6,
                    name="shuffled views",
                    legendgroup="shuf",
                    showlegend=f is shuf[0],
                    hovertemplate="shuffled<br>λ = %{x}<br>gain = %{y:.3f}<extra></extra>",
                )
            )
        for name, f, color in (
            ("log views", views, BLUE),
            ("log account age", age, ORANGE),
            ("log degree", deg, AQUA),
        ):
            k = f["n_flow_basins"]
            fig.add_trace(
                go.Scatter(
                    x=[c["lam"] for c in f["coarse"]],
                    y=[
                        c["macro_ei"] / np.log2(k) - c["micro_ei"] / np.log2(net["n"])
                        for c in f["coarse"]
                    ],
                    mode="lines+markers",
                    name=f"{name} ({k} flow basins)",
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"{name}<br>λ = %{{x}}<br>gain = %{{y:.3f}}<extra></extra>",
                )
            )
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        fig.update_layout(
            height=380,
            xaxis_title="mass λ",
            yaxis_title="macro − micro effectiveness",
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig, key="tw_coarse", width="stretch", theme="streamlit")
        best_ce = max(c["macro_ei"] - c["micro_ei"] for f in fields for c in f["coarse"])
        adv_v = [c["macro_ei"] - c["null_ei"] for c in views["coarse"]]
        st.caption(
            "As on the synthetic graphs, the basin scale overtakes the streamer scale per state "
            "once messages are heavy (above 0), and strict causal emergence never appears (the "
            f"closest is {-best_ce:.1f} bits short). Real views' {views['n_flow_basins']} flow "
            f"basins beat random groupings of the same sizes by {adv_v[0]:.1f} bits at λ = 0 "
            f"and {adv_v[-1]:.1f} at λ = {views['coarse'][-1]['lam']:g}."
        )

        st.subheader("What it means")
        st.markdown(
            "- **Real popularity is arranged, not random.** Popular streamers befriend each "
            "other, so the landscape has a few large, connected deep basins instead of hundreds "
            "of scattered ones. That one fact drives most of what follows.\n"
            "- **The best mass and the glassy slowdown carry over.** The U-shaped hitting time "
            "appears with a best mass near 2, and heavy messages condense into one basin almost "
            "completely, with the rise starting just above the trap model's λ_c.\n"
            "- **Mass sorting does not.** With one dominant core, almost every message, light "
            "or heavy, ends up in the same place. Shuffling who is popular brings sorting back.\n"
            "- **Coarse-graining behaves as on synthetic graphs:** per state the basin scale "
            "wins for heavy messages, but never in strict bits.\n"
            "- **Limits:** one network, one snapshot (May 2018), English-language streamers "
            "only. View counts are lifetime totals, which also grow with account age."
        )
with tab_coarse:
    st.header("Does coarse-graining help heavy messages?")
    crs = coarse()
    if crs is None:
        empty_state("No results yet. Run the coarse-graining experiment to fill this section.")
    else:
        C = coarse_frame()
        cl = np.array(crs["config"]["lams"])
        cmouse = {m["field"]: m for m in crs["mouse"]}

        st.subheader("The question")
        st.write(
            "The causal-emergence entry asked whether grouping nodes into *macro-nodes* gives "
            "a more informative description of a random walk than the nodes themselves. Here "
            "the walk is the gravity walk, and the macro-nodes come from the landscape. The "
            "question is whether the same grouping becomes more informative as messages get "
            "heavier."
        )

        st.subheader("How I measured it")
        st.markdown(
            "- **Flow basins:** in the heavy-mass limit every step goes to the deepest "
            "neighbour. Following that rule from any node ends on a pair of nodes that are each "
            "other's deepest neighbour, where a heavy message bounces. Each such pair, with "
            "every node that flows into it, is one macro-node. The grouping depends only on "
            "the landscape, so the same macro-nodes are scored at every mass.\n"
            "- **Macro walk:** a macro-node's next step is the average of its members' next "
            "steps, weighted by how much time the walk spends at each member, π(λ).\n"
            "- **Effective information (EI)**, in bits, for the micro walk and the macro walk. "
            "Strict *causal emergence* means macro EI is higher than micro EI.\n"
            "- **Effectiveness:** EI divided by its maximum, log₂ of the number of states. It "
            "compares scales of different sizes on the same 0–1 footing.\n"
            f"- **Control:** {crs['config']['n_null']} random groupings with the same group "
            "sizes, for each setting and mass.\n"
            f"- **Settings:** BA and ER graphs (n = 1,000), the ρ dial at "
            f"{', '.join(f'{p:g}' for p in crs['config']['fields']['rho'])} and heat kernel t = "
            f"{', '.join(f'{p:g}' for p in crs['config']['fields']['heat'])}, with "
            f"{crs['config']['seeds']} seeds each, plus the mouse connectome."
        )

        st.subheader("Flow basins beat random groupings, most at moderate mass")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Barabási–Albert", "Erdős–Rényi"))
        rho_colors = {0.0: BLUE, 0.5: ORANGE, 1.0: AQUA}
        for col, gname in enumerate(("BA", "ER"), start=1):
            for par, color in rho_colors.items():
                sub = C[(C["graph"] == gname) & (C["kind"] == "rho") & (C["par"] == par)]
                agg = sub.groupby("lam")[["micro_ei", "macro_ei", "null_ei"]].mean()
                for metric, dash, label in (
                    ("micro_ei", "dot", "nodes"),
                    ("macro_ei", "solid", "flow basins"),
                    ("null_ei", "dash", "random groups"),
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=agg.index,
                            y=agg[metric],
                            mode="lines",
                            name=f"ρ = {par:g}, {label}",
                            legendgroup=f"{par}{label}",
                            showlegend=col == 1,
                            line=dict(color=color, dash=dash, width=2),
                            hovertemplate=f"ρ = {par:g}, {label}<br>λ = %{{x}}<br>EI = "
                            "%{y:.2f} bits<extra></extra>",
                        ),
                        row=1,
                        col=col,
                    )
        fig.update_xaxes(title_text="mass λ")
        fig.update_yaxes(title_text="EI (bits)", col=1)
        fig.update_layout(height=440, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, key="coarse_ei", width="stretch", theme="streamlit")
        adv = C[C["k"] > 1].groupby("lam")["advantage"].mean()
        st.caption(
            "Mean over seeds. Dotted: the micro walk. Solid: flow basins. Dashed: random "
            "groupings of the same sizes. The gap between solid and dashed is what the "
            "landscape's grouping adds. Averaged over all settings with more than one basin, "
            f"it is {adv.iloc[0]:.2f} bits for massless messages (neighbours share a basin, so "
            f"flow stays inside even without gravity), peaks at {adv.max():.2f} bits near "
            f"λ = {adv.idxmax():g}, and is {adv.iloc[-1]:.2f} at λ = {cl[-1]:g}. When gravity "
            "follows degree on BA (ρ = 1) the whole graph drains into one basin, so there is a "
            "single macro-node and nothing to compare."
        )

        st.subheader("Strict causal emergence: no")
        best_ce = C.groupby(["graph", "kind", "par", "seed"])["ce"].max()
        multi = C[C["k"] > 1]
        st.write(
            "Macro EI never exceeds micro EI here: the closest any setting and mass comes is "
            f"{-best_ce.max():.2f} bits short. The macro scale has far fewer states "
            f"(median {int(multi.groupby(['graph', 'kind', 'par', 'seed'])['k'].first().median())} "
            "flow basins against about 1,000 nodes), and EI is capped at log₂ of the number of "
            "states, so coarse-graining loses more bits of capacity than it gains in sharpness."
        )

        st.subheader("Per state, the macro scale overtakes the micro scale")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Barabási–Albert", "Erdős–Rényi"))
        field_colors = {
            ("rho", 0.0): BLUE,
            ("rho", 0.25): "#5b93dd",
            ("rho", 0.5): ORANGE,
            ("rho", 0.75): "#f19a6d",
            ("rho", 1.0): AQUA,
            ("heat", 0.5): PURPLE,
            ("heat", 2.0): "#b495e6",
        }
        for col, gname in enumerate(("BA", "ER"), start=1):
            for (kind, par), color in field_colors.items():
                sub = C[
                    (C["graph"] == gname) & (C["kind"] == kind) & (C["par"] == par) & (C["k"] > 1)
                ]
                if sub.empty:
                    continue
                agg = sub.groupby("lam")[["eff_gain", "null_eff_gain"]].mean()
                name = f"ρ = {par:g}" if kind == "rho" else f"heat t = {par:g}"
                fig.add_trace(
                    go.Scatter(
                        x=agg.index,
                        y=agg["eff_gain"],
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=col == 1,
                        line=dict(color=color, width=2),
                        hovertemplate=f"{name}<br>λ = %{{x}}<br>gain = %{{y:.3f}}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
            null = C[(C["graph"] == gname) & (C["k"] > 1)].groupby("lam")["null_eff_gain"].mean()
            fig.add_trace(
                go.Scatter(
                    x=null.index,
                    y=null,
                    mode="lines",
                    name="random groups (all fields)",
                    legendgroup="null",
                    showlegend=col == 1,
                    line=dict(color=GRAY, dash="dash", width=2),
                    hovertemplate="random groups<br>λ = %{x}<br>gain = %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=col,
            )
            fig.add_hline(y=0, line=dict(color=GRAY, width=1), row=1, col=col)
        fig.update_xaxes(title_text="mass λ")
        fig.update_yaxes(title_text="macro − micro effectiveness", col=1)
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, key="coarse_eff", width="stretch", theme="streamlit")
        cross = coarse_crossings(C)
        st.caption(
            "Effectiveness is EI ÷ log₂(states), so 1 means every state determines a different "
            "next state. Above 0, each flow basin carries more of its possible information than "
            f"each node does. Flow basins overtake the nodes in {cross['crossed']} of "
            f"{cross['total']} settings with more than one basin, at a median mass of "
            f"λ = {cross['median']:g}. Random groupings of the same sizes overtake them in "
            f"{cross['null_crossed']}, so the gain comes from the grouping, not from having "
            "fewer states."
        )

        st.subheader("Why: determinism up, degeneracy gone")
        rough = C[(C["graph"] == "BA") & (C["kind"] == "rho") & (C["par"] == 0.0)]
        parts = rough.groupby("lam")[["micro_det", "micro_deg", "macro_det", "macro_deg"]].mean()
        lam_lo, lam_hi = cl[0], cl[-1]
        st.dataframe(
            pd.DataFrame(
                {
                    "scale": ["nodes", "nodes", "flow basins", "flow basins"],
                    "mass": [lam_lo, lam_hi, lam_lo, lam_hi],
                    "determinism": [
                        parts.loc[lam_lo, "micro_det"],
                        parts.loc[lam_hi, "micro_det"],
                        parts.loc[lam_lo, "macro_det"],
                        parts.loc[lam_hi, "macro_det"],
                    ],
                    "degeneracy": [
                        parts.loc[lam_lo, "micro_deg"],
                        parts.loc[lam_hi, "micro_deg"],
                        parts.loc[lam_lo, "macro_deg"],
                        parts.loc[lam_hi, "macro_deg"],
                    ],
                }
            ).assign(ei=lambda d: d["determinism"] - d["degeneracy"]),
            hide_index=True,
            column_config={
                "scale": "Scale",
                "mass": st.column_config.NumberColumn("Mass λ", format="%g"),
                "determinism": st.column_config.NumberColumn("Determinism (bits)", format="%.2f"),
                "degeneracy": st.column_config.NumberColumn("Degeneracy (bits)", format="%.2f"),
                "ei": st.column_config.NumberColumn("EI (bits)", format="%.2f"),
            },
        )
        st.caption(
            "Rough BA fields (ρ = 0), mean over seeds. EI = determinism − degeneracy. Heavy "
            "messages make each node's next step sharp, but many nodes then point at the same "
            "few well bottoms (degeneracy). Grouping a basin into one macro-node turns that "
            "shared destination into a self-loop: at λ = "
            f"{lam_hi:g} a flow basin keeps "
            f"{rough[rough['lam'] == lam_hi]['self_loop'].mean():.0%} of its flow on average."
        )

        st.subheader("The mouse brain")
        st.dataframe(
            pd.DataFrame(
                [
                    dict(
                        field={
                            "log volume": "Region volume (log)",
                            "FA": "Fractional anisotropy",
                        }.get(k, "Random (control)"),
                        k=m["k"],
                        micro0=m["rows"][0]["micro_ei"] / np.log2(m["n"]),
                        macro0=m["rows"][0]["macro_ei"] / np.log2(m["k"]) if m["k"] > 1 else None,
                        micro8=m["rows"][-1]["micro_ei"] / np.log2(m["n"]),
                        macro8=m["rows"][-1]["macro_ei"] / np.log2(m["k"]) if m["k"] > 1 else None,
                        adv8=m["rows"][-1]["macro_ei"] - m["rows"][-1]["null_ei"],
                    )
                    for k, m in cmouse.items()
                ]
            ),
            hide_index=True,
            column_config={
                "field": "Gravity",
                "k": st.column_config.NumberColumn("Flow basins", format="%d"),
                "micro0": st.column_config.NumberColumn("Nodes, λ = 0", format="%.2f"),
                "macro0": st.column_config.NumberColumn("Basins, λ = 0", format="%.2f"),
                "micro8": st.column_config.NumberColumn(f"Nodes, λ = {cl[-1]:g}", format="%.2f"),
                "macro8": st.column_config.NumberColumn(f"Basins, λ = {cl[-1]:g}", format="%.2f"),
                "adv8": st.column_config.NumberColumn(
                    f"Beats random by (bits, λ = {cl[-1]:g})", format="%.2f"
                ),
            },
        )
        st.caption(
            "Effectiveness of each scale on the 332-region backbone. The node-to-region mapping "
            "behind volume and FA is inferred, not documented."
        )

        st.subheader("What it means")
        st.markdown(
            "- **Strict causal emergence: no.** With about 1,000 nodes and tens of basins, the "
            "macro walk always has less EI than the micro walk; the closest any setting "
            f"comes is {-best_ce.max():.1f} bits short.\n"
            "- **Per state, heavy messages are best described basin by basin.** Above "
            f"λ ≈ {cross['median']:g} each flow basin is more effective than each node, in "
            "every setting with more than one basin and on the mouse brain, and random "
            "groupings never are. For "
            "massless messages the node scale wins.\n"
            "- **So the answer depends on the yardstick.** Counted in bits, coarse-graining "
            "never pays. Counted per state, it starts to pay once messages are heavy enough to "
            'follow the landscape, which is the version of "more informative for heavy '
            'messages" that holds here.\n'
            "- **Limits:** one grouping rule (flow basins). The greedy EI search used in the "
            "causal-emergence entry is too slow at 1,000 nodes, so a better grouping may exist. "
            "Mass only goes up to λ = 8."
        )

st.header("Code")
st.markdown(" · ".join(f"[{p}]({REPO_URL}/blob/main/{p})" for p in CODE_PATHS))
