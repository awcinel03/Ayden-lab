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
CODE_PATHS = (
    "lab/gravity_mass/model.py",
    "pipelines/experiments/network_science/gravity_mass_glass.py",
)
# Categorical slots 1-4 (validated palette); network sizes use one sequential blue ramp.
BLUE, ORANGE, AQUA, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8e5cd9", "#8a8a85"
FIELD_COLORS = {"gauss": BLUE, "exp_deep": ORANGE, "exp_shallow": AQUA, "laplace": PURPLE}
SHORT = {
    "gauss": "Gaussian",
    "exp_deep": "Exponential, deep tail",
    "exp_shallow": "Exponential, shallow tail",
    "laplace": "Laplace",
}
SIZE_RAMP = ["#a9c9f2", "#6fa3e6", "#2a78d6", "#1a4f94"]


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
            "- **Four shapes of gravity**, all standardized: Gaussian, exponential with its long "
            "tail toward deep wells, exponential with its long tail toward shallow ground, and "
            "Laplace (both tails exponential).\n"
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
                "field": "Gravity shape",
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
            "- **Condensation happens, and it needs no special landscape.** In all four shapes "
            "the finite-size slope moves from about −1 to about 0 over a narrow window of mass: "
            "finite-size estimates of λ_c run from "
            f"{min(fs.values()):.1f} to {max(fs.values()):.1f}. For Gaussian gravity the slope is "
            f"{slope_at('gauss', 2):.2f} at λ = 2 and {slope_at('gauss', 6):.2f} at λ = 6. Above "
            "the window, the share of time held by the deepest few wells no longer shrinks with "
            "network size. That is the signature of a glass-like, condensed phase.\n"
            "- **Exponential climbs come for free.** Even with Gaussian gravity, trap climbs "
            "have a roughly exponential tail. A climb is a difference between edge depths "
            "(floor minus rim), and extremes of such differences tend toward exponential tails. "
            "So the trap model applies without building it in.\n"
            "- **For Gaussian gravity the trap model gets the critical mass right:** λ_c = 1/Γ₀ = "
            f"{F['gauss']['lam_pred']:.1f} against {F['gauss']['lam_fs']:.1f} from finite-size "
            "scaling.\n"
            "- **For the other shapes it does not.** Their climb tails are not cleanly "
            "exponential: the mean excess drifts with the threshold (down for the deep and "
            "Laplace shapes, up for the shallow one). A single Γ₀ then misses: "
            + "; ".join(
                f"{SHORT[k]} {F[k]['lam_pred']:.1f} predicted, {F[k]['lam_fs']:.1f} measured"
                for k in ("exp_deep", "exp_shallow", "laplace")
            )
            + ". Measuring Γ₀ only on the deepest traps narrows the gap but does not close it "
            "for every shape at once, so I report the estimator fixed before the full run.\n"
            "- **Condensation is weaker than the trap model expects.** At λ = 10 and n = 4,000, "
            f"Y₂ is {F['gauss']['mean'][-1][-1]:.2f} for Gaussian gravity against "
            f"{1 - F['gauss']['lam_pred'] / lams[-1]:.2f} predicted. Time stays split across a few "
            "wells rather than one.\n"
            "- **Limits:** sizes span a factor of 8 (500 to 4,000 nodes). That shows a clear "
            "crossover but cannot prove a true transition in the infinite-size limit. The "
            "networks are Erdős–Rényi only."
        )

with tab_sort:
    st.header("Mass sorting")
    empty_state(
        "Next up. Release messages of many masses from the same starts and measure how much "
        "a message's mass tells you about which deep well it reaches first."
    )
with tab_social:
    st.header("A social network")
    empty_state(
        "Planned: Twitch's mutual-friendship networks (Rozemberczki et al.), with each "
        "streamer's view count as well depth."
    )
with tab_coarse:
    st.header("Coarse-graining")
    empty_state(
        "Planned: does grouping nodes into watersheds (each node with the well it drains "
        "into) raise effective information more for heavy messages than for light ones?"
    )

st.header("Code")
st.markdown(" · ".join(f"[{p}]({REPO_URL}/blob/main/{p})" for p in CODE_PATHS))
