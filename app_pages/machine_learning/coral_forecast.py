import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import db, storage
from lab.registry import entry
from lab.ui import empty_state, page_header, provenance_footer

ENTRY = entry("coral-heat-stress-forecast")
CODE_PATHS = (
    "pipelines/ingest/crw_erddap.py",
    "pipelines/experiments/machine_learning/coral_forecast.py",
)
# Categorical slots 1-2 (validated palette), assigned in fixed order.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
# Ordinal blue ramp (steps 250/400/500/650): one hue that darkens with severity.
RISK = {"Low": "#86b6ef", "Moderate": "#3987e5", "High": "#256abf", "Extreme": "#104281"}
FEATURE_SETS = {
    "sst_ssta": "SST + SSTA lags",
    "lags": "+ DHW lags",
    "full": "+ rate-of-change",
}

page_header(ENTRY)

st.header("Purpose")
st.write(
    "Coral bleaches when the water stays too warm for too long. NOAA Coral Reef Watch "
    "measures that accumulated heat as **Degree Heating Weeks (DHW)**: at 4 °C-weeks "
    "significant bleaching becomes likely, and at 8 it turns lethal. I wanted to know whether a "
    "model could see that stress coming four weeks before it arrives on the Great Barrier "
    "Reef, early enough for managers to act."
)
st.write(
    "I started with a classifier for NOAA's bleaching alert level, then dropped it. The alert "
    "level isn't an independent observation: NOAA computes it from same-day DHW and "
    "HotSpot with fixed thresholds, so a classifier fed DHW mostly relearns that rule. "
    "Forecasting a *future* DHW removes the circularity, because the target doesn't exist yet "
    "when the prediction is made."
)

st.header("Method")
st.markdown(
    "- **Data:** NOAA Coral Reef Watch v3.1 daily 5 km sea surface temperature (SST), SST "
    "anomaly (SSTA) and DHW for the Great Barrier Reef (24.5°S–10°S, 142°E–154°E), "
    "2005–2023, every 20th grid cell (~1°), via PacIOOS ERDDAP.\n"
    "- **Features (15):** SST, SSTA and DHW today and 7, 14 and 28 days earlier, plus three "
    "differences: the 7- and 14-day SST trend and the 7-day DHW acceleration. A Random "
    "Forest splits one feature at a time, so it can't compute a difference itself.\n"
    "- **Target:** DHW 28 days after the feature date.\n"
    "- **No leakage:** each cell is put on a complete daily calendar before lagging, so a "
    "missing day becomes a gap rather than a silently longer lag. The split is by time "
    "(train 2005–2019, test 2020–2023), and training rows whose target falls inside the test "
    "window are dropped (an embargo).\n"
    "- **Model:** scikit-learn Random Forest, 200 trees, minimum 5 samples per leaf, seed 0.\n"
    "- **Baselines:** always predict the training mean; and *persistence*, DHW in 28 days = "
    "DHW today. Persistence is strong because DHW is a slow 12-week accumulator, and today's "
    "DHW is itself a feature, so beating it is the floor, not the achievement.\n"
    "- **Timing:** on the reef-wide daily mean, the shift (in days) that best aligns forecast "
    "and reality, and the first date each summer (Jan–Jun) that each crosses DHW 4."
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
    v = scalars.get(metric)
    return float("nan") if v is None or pd.isna(v) else float(v)


def series(metric: str, *dims: str) -> pd.DataFrame:
    """Rows of ``metric`` with the requested dims expanded into columns."""
    rows = results[results["metric"] == metric]
    out = pd.DataFrame({d: rows["dims"].map(lambda x, d=d: x.get(d)) for d in dims})
    out["value"] = rows["value"].to_numpy()
    return out.reset_index(drop=True)


def show(fig: go.Figure, key: str, height: int = 380) -> None:
    """Render a Plotly figure with the lab's shared layout defaults."""
    fig.update_layout(height=height, margin={"l": 0, "r": 0, "t": 30, "b": 0})
    st.plotly_chart(fig, key=key, width="stretch", theme="streamlit")


arts = {a.storage_path.rsplit("/", 1)[-1]: a for a in db.artifacts(run.id)}
daily = storage.read_parquet(arts["daily.parquet"]) if "daily.parquet" in arts else None
cells = storage.read_parquet(arts["cells.parquet"]) if "cells.parquet" in arts else None

st.header("Results")
with st.container(horizontal=True):
    st.metric("RMSE", f"{value('rmse'):.3f} °C-wk", border=True)
    st.metric("R²", f"{value('r2'):.3f}", border=True)
    st.metric("Better than persistence", f"{value('skill_vs_persistence'):+.1%}", border=True)
    st.metric("Mean warning lateness", f"{value('mean_late_days'):+.1f} days", border=True)
st.caption(
    f"Test window 2020–2023: {value('n_test'):,.0f} cell-days; trained on "
    f"{value('n_train'):,.0f} ({value('embargoed_rows'):,.0f} dropped by the embargo), from "
    f"{value('grid_cells'):,.0f} grid cells. Neighbouring cells are strongly correlated, so "
    "these are far fewer independent observations than the counts suggest."
)

st.subheader("Forecast vs reality")
if daily is None or daily.empty:
    empty_state("The daily series is not available for this run.")
else:
    fig = go.Figure()
    for column, name, color, dash in (
        ("actual", "Actual DHW", BLUE, "solid"),
        ("predicted", "Forecast (28 days earlier)", ORANGE, "solid"),
        ("persistence", "Persistence baseline", GRAY, "dot"),
    ):
        fig.add_trace(
            go.Scatter(
                x=daily["target_date"],
                y=daily[column],
                name=name,
                mode="lines",
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
            )
        )
    for level, label in ((4, "Bleaching likely (4)"), (8, "Mortality likely (8)")):
        fig.add_hline(
            y=level,
            line={"color": GRAY, "width": 1, "dash": "dash"},
            annotation_text=label,
            annotation_position="top right",
        )
    fig.update_layout(
        hovermode="x unified",
        yaxis_title="Reef-wide mean DHW (°C-weeks)",
        legend={"orientation": "h", "y": 1.12},
    )
    show(fig, "forecast", height=420)
    st.caption(
        "Reef-wide daily mean over the sampled grid cells, plotted on the date being "
        "forecast. Persistence is what you'd get by assuming nothing changes for four weeks."
    )

left, right = st.columns(2)
with left:
    st.markdown("**Error against the baselines**")
    bars = pd.DataFrame(
        {
            "model": ["Training mean", "Persistence", "Random Forest"],
            "rmse": [
                value("rmse_baseline_mean"),
                value("rmse_baseline_persistence"),
                value("rmse"),
            ],
        }
    )
    fig = go.Figure(
        go.Bar(
            x=bars["rmse"],
            y=bars["model"],
            orientation="h",
            marker_color=[GRAY, GRAY, BLUE],
            text=bars["rmse"].map("{:.3f}".format),
            textposition="outside",
            hovertemplate="%{y}: RMSE %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="RMSE (°C-weeks, lower is better)")
    show(fig, "baselines", height=260)
with right:
    st.markdown("**What the model relies on**")
    imp = series("feature_importance", "feature").sort_values("value")
    fig = go.Figure(
        go.Bar(
            x=imp["value"],
            y=imp["feature"],
            orientation="h",
            marker_color=BLUE,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Impurity importance (sums to 1)")
    show(fig, "importance", height=420)

st.subheader("How the feature set got here")
ab = {
    m: series(f"ablation_{m}", "feature_set").set_index("feature_set")["value"]
    for m in ("n_features", "rmse", "r2", "skill_vs_persistence", "best_shift", "mean_late")
}
table = pd.DataFrame(ab).reindex([k for k in FEATURE_SETS if k in ab["rmse"].index])
if not table.empty:
    st.dataframe(
        pd.DataFrame(
            {
                "Features": [
                    f"{FEATURE_SETS[k]} ({n:.0f})" for k, n in table["n_features"].items()
                ],
                "RMSE": table["rmse"].round(3),
                "R²": table["r2"].round(3),
                "vs persistence": table["skill_vs_persistence"].map("{:+.1%}".format),
                "Best shift (days)": table["best_shift"].map("{:+.0f}".format),
                "Mean lateness (days)": table["mean_late"].map(
                    lambda v: "–" if pd.isna(v) else f"{v:+.1f}"
                ),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("Same split, same model, same rows; only the features change.")

st.subheader("Is the warning on time?")
left, right = st.columns(2)
with left:
    curve = series("shift_rmse", "shift").sort_values("shift")
    best = value("best_shift_days")
    fig = go.Figure(
        go.Scatter(
            x=curve["shift"],
            y=curve["value"],
            mode="lines+markers",
            line={"color": BLUE, "width": 2},
            marker={"size": 8},
            hovertemplate="Shift %{x:+d} days: RMSE %{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": GRAY, "width": 1, "dash": "dash"})
    fig.update_layout(
        xaxis_title="Shift applied to the forecast (days; negative = earlier)",
        yaxis_title="RMSE of the daily mean",
    )
    show(fig, "shift", height=300)
    st.caption(
        f"The forecast fits best moved {abs(best):.0f} days earlier "
        f"(RMSE {value('rmse_at_best_shift'):.3f} vs {value('rmse_unshifted'):.3f} as issued)."
    )
with right:
    cross = series("crossing_late_days", "year", "actual", "predicted").sort_values("year")

    def verdict(r: pd.Series) -> str:
        if pd.notna(r["value"]):
            return f"{r['value']:+.0f} days"
        return "false alarm" if r["predicted"] else "never crossed" if not r["actual"] else "missed"

    st.markdown("**First crossing of DHW 4, each summer**")
    st.dataframe(
        pd.DataFrame(
            {
                "Summer": cross["year"].astype(int).astype(str),
                "Actual": cross["actual"].fillna("never"),
                "Forecast": cross["predicted"].fillna("never"),
                "Forecast was": cross.apply(verdict, axis=1),
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("Risk map")
if cells is None or cells.empty:
    empty_state("Per-cell predictions are not available for this run.")
else:
    cells["target_date"] = pd.to_datetime(cells["target_date"])
    by_day = cells.groupby("target_date")["predicted"].mean()
    dates = by_day.index.date.tolist()
    worst = by_day.idxmax().date()
    day = st.select_slider(
        "Forecast for",
        options=dates,
        value=worst,
        format_func=lambda d: d.strftime("%d %b %Y"),
        key="map_date",
    )
    snap = cells[cells["target_date"].dt.date == day].copy()
    snap["band"] = pd.cut(
        snap["predicted"].clip(lower=0),
        bins=[-0.1, 1, 4, 8, float("inf")],
        labels=list(RISK),
        right=False,
    )
    fig = go.Figure()
    for band, color in RISK.items():
        part = snap[snap["band"] == band]
        fig.add_trace(
            go.Scattermap(
                lat=part["latitude"],
                lon=part["longitude"],
                mode="markers",
                name=band,
                marker={"size": 14, "color": color},
                customdata=part[["predicted", "actual"]].to_numpy(),
                hovertemplate=(
                    f"<b>{band}</b><br>Forecast DHW %{{customdata[0]:.1f}}<br>"
                    "Actual DHW %{customdata[1]:.1f}<extra></extra>"
                ),
            )
        )
    dark = st.context.theme.type == "dark"
    fig.update_layout(
        map={
            "style": "carto-darkmatter" if dark else "carto-positron",
            "center": {"lat": -17.25, "lon": 148.0},
            "zoom": 4.3,
        },
        legend={"orientation": "h", "y": 1.06, "title": {"text": "Forecast risk"}},
    )
    show(fig, "risk_map", height=560)
    st.caption(
        "One marker per ~1° grid cell, coloured by forecast DHW using NOAA's bands: Low < 1, "
        "Moderate 1–4, High 4–8, Extreme ≥ 8. It opens on the test window's worst forecast "
        "day; an average day is almost all Low."
    )

st.header("Takeaways")
imp_rank = series("feature_importance", "feature").sort_values("value", ascending=False)
top = imp_rank.iloc[0] if not imp_rank.empty else None
second = imp_rank.iloc[1] if len(imp_rank) > 1 else None
late = cross.dropna(subset=["value"]) if not cross.empty else cross
bullets = [
    f"**It beats both baselines.** RMSE {value('rmse'):.3f} against "
    f"{value('rmse_baseline_persistence'):.3f} for persistence and "
    f"{value('rmse_baseline_mean'):.3f} for the mean (R² {value('r2'):.3f}). Because today's "
    "DHW is a feature, matching persistence was free. The real gain is the "
    f"{value('skill_vs_persistence'):.0%} beyond it.",
]
if top is not None and second is not None:
    bullets.append(
        f"**Persistence, adjusted for momentum.** `{top['feature']}` carries "
        f"{top['value']:.0%} of the importance and `{second['feature']}` {second['value']:.0%}. "
        "I read the model as 'today's heat stress, corrected for how fast it is building'."
    )
if "lags" in table.index and "sst_ssta" in table.index:
    gain = table.at["full", "skill_vs_persistence"] - table.at["lags", "skill_vs_persistence"]
    bullets.append(
        f"**DHW lags were the big win.** On temperature alone the forest only drew level with "
        f"persistence ({table.at['sst_ssta', 'skill_vs_persistence']:+.1%}). Adding DHW lags took "
        f"it to {table.at['lags', 'skill_vs_persistence']:+.1%}; the rate-of-change terms added "
        f"a further {gain * 100:.1f} percentage points."
    )
if not late.empty:
    bullets.append(
        f"**But it warns late, the failure that matters for early warning.** It fits best "
        f"shifted {abs(value('best_shift_days')):.0f} days earlier, and in the "
        f"{len(late)} summers where both crossed DHW 4 it was late every time "
        f"(mean {value('mean_late_days'):+.1f} days). R² hides this completely."
        if (late["value"] > 0).all()
        else f"**Timing is mixed.** Mean lateness at DHW 4 is {value('mean_late_days'):+.1f} days."
    )
if "lags" in table.index:
    bullets.append(
        f"**The rate-of-change features barely fixed it.** Mean lateness went from "
        f"{table.at['lags', 'mean_late']:+.1f} to {table.at['full', 'mean_late']:+.1f} days. "
        "I think the reason is structural: DHW is itself a backward-looking 12-week sum, so "
        "every feature built from it inherits the lag. Real early warning probably needs "
        "forward-looking inputs such as NOAA's four-month SST outlook, which I haven't tested."
    )
if "sst_ssta" in table.index and table.at["sst_ssta", "mean_late"] < value("mean_late_days"):
    bullets.append(
        f"**A hint worth following up.** The temperature-only model is much less accurate but "
        f"less late: best shift {table.at['sst_ssta', 'best_shift']:+.0f} days, mean lateness "
        f"{table.at['sst_ssta', 'mean_late']:+.1f} days. That fits the idea that the DHW inputs "
        "carry the lag, though these are three or four summers on correlated grid cells, so I "
        "treat it as a lead, not a result."
    )
st.markdown("\n".join(f"- {b}" for b in bullets))

provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)
