"""Forecasting Great Barrier Reef heat stress (DHW) 28 days ahead with a Random Forest.

Run: python -m pipelines.experiments.machine_learning.coral_forecast [--allow-dirty]

A port of the forecasting extension of my coral_bleaching project. Builds lag and
rate-of-change features on a complete daily calendar per grid cell, splits by
time with an embargo at the train/test boundary, fits the Random Forest, scores
it against mean and persistence baselines, measures forecast timing (best
shift, DHW 4 threshold crossings), and repeats the fit for the README's
feature-set ablation. Writes long-format results plus two Parquet artifacts:
the regional daily mean series and per-cell test predictions for the risk map.
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.common.client import get_client
from pipelines.common.provenance import git_sha, sha256_bytes
from pipelines.common.runs import record_artifact, start_run, tracked, write_results
from pipelines.common.storage import upload
from pipelines.common.sync import sync_experiments
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "coral-heat-stress-forecast"
DATASET_SLUG = "crw-dhw-gbr-2005-2023-stride20"
ARTIFACT_BUCKET = "artifacts"
RENAME = {"CRW_SST": "sst", "CRW_SSTANOMALY": "ssta", "CRW_DHW": "dhw"}
LAGS = (0, 7, 14, 28)
DIFFS = (
    ("sst_trend7", "sst_lag0", "sst_lag7"),
    ("sst_trend14", "sst_lag0", "sst_lag14"),
    ("dhw_accel7", "dhw_lag0", "dhw_lag7"),
)
CONFIG: dict[str, Any] = {
    "version": 1,
    "datasets": [DATASET_SLUG],
    "horizon_days": 28,
    "lags_days": list(LAGS),
    "lag_variables": ["sst", "ssta", "dhw"],
    "diffs": [list(d) for d in DIFFS],
    "panel": "every cell reindexed onto a complete daily calendar before shifting",
    "train_years": [2005, 2019],
    "test_years": [2020, 2023],
    "embargo": "drop training rows whose target date is on or after the first test day",
    "model": {
        "type": "sklearn RandomForestRegressor",
        "n_estimators": 200,
        "min_samples_leaf": 5,
        "max_depth": None,
        "random_state": 0,
        "n_jobs": -1,
    },
    "baselines": ["training mean", "persistence (DHW in 28 days = DHW today)"],
    "timing": {"threshold_dhw": 4.0, "max_shift_days": 21, "season": "Jan 1 - Jun 30"},
    # README "How the feature set got here": same split and model, only the features change.
    "ablation": {
        "sst_ssta": {"variables": ["sst", "ssta"], "diffs": False},
        "lags": {"variables": ["sst", "ssta", "dhw"], "diffs": False},
        "full": {"variables": ["sst", "ssta", "dhw"], "diffs": True},
    },
    "risk_bands": [["Low", 0, 1], ["Moderate", 1, 4], ["High", 4, 8], ["Extreme", 8, None]],
}
TARGET = f"dhw_plus{CONFIG['horizon_days']}d"


def feature_columns(variables: list[str], diffs: bool) -> list[str]:
    """Lag columns for ``variables``, then the rate-of-change columns whose sources exist."""
    lags = [f"{v}_lag{lag}" for v in variables for lag in LAGS]
    if not diffs:
        return lags
    return lags + [name for name, a, b in DIFFS if a in lags and b in lags]


def complete_daily_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Every (cell, day) in the date range; missing days become NaN rows, so a positional
    shift within a cell is a shift in days."""
    cells = frame[["latitude", "longitude"]].drop_duplicates()
    dates = pd.DataFrame({"date": pd.date_range(frame["date"].min(), frame["date"].max())})
    panel = cells.merge(dates, how="cross").merge(
        frame, on=["latitude", "longitude", "date"], how="left"
    )
    return panel.sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)


def build_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Lag and rate-of-change features plus the DHW target ``horizon`` days ahead.

    Rows lacking any full-model feature or the target are dropped, so every
    ablation trains and tests on the same rows.
    """
    frame = frame.rename(columns=RENAME)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    panel = complete_daily_panel(frame[["date", "latitude", "longitude", "sst", "ssta", "dhw"]])
    cells = panel.groupby(["latitude", "longitude"], sort=False)
    derived = {
        f"{v}_lag{lag}": panel[v] if lag == 0 else cells[v].shift(lag)
        for v in ("sst", "ssta", "dhw")
        for lag in LAGS
    }
    derived[TARGET] = cells["dhw"].shift(-horizon)
    panel = panel.assign(**derived)
    panel = panel.assign(**{name: panel[a] - panel[b] for name, a, b in DIFFS})
    panel["target_date"] = panel["date"] + pd.Timedelta(days=horizon)
    full = feature_columns(["sst", "ssta", "dhw"], diffs=True)
    columns = ["date", "target_date", "latitude", "longitude", *full, "dhw", TARGET]
    return panel[columns].dropna(subset=[*full, TARGET]).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, train_years: list[int], test_years: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Train and test by calendar year, with the embargo; also returns rows embargoed."""
    year = frame["date"].dt.year
    train = frame[(year >= train_years[0]) & (year <= train_years[1])]
    test = frame[(year >= test_years[0]) & (year <= test_years[1])]
    keep = train["target_date"] < pd.Timestamp(f"{test_years[0]}-01-01")
    return train[keep].reset_index(drop=True), test.reset_index(drop=True), int((~keep).sum())


def rmse(truth: np.ndarray, guess: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(truth, guess)))


def timing(
    daily: pd.DataFrame, threshold: float, max_shift: int
) -> tuple[dict[int, float], pd.DataFrame]:
    """RMSE of the regional daily mean at each shift, and each summer's first threshold crossing.

    ``daily`` is indexed by target date with ``actual`` and ``predicted``. A negative
    best shift means the forecast lags; positive ``late_days`` means it warned late.
    """
    actual, predicted = daily["actual"].to_numpy(), daily["predicted"].to_numpy()

    def shifted(shift: int) -> float:
        if shift < 0:
            a, b = predicted[-shift:], actual[:shift]
        elif shift > 0:
            a, b = predicted[:-shift], actual[shift:]
        else:
            a, b = predicted, actual
        return float(np.sqrt(((a - b) ** 2).mean()))

    curve = {s: shifted(s) for s in range(-max_shift, max_shift + 1)}
    records = []
    for year in sorted(daily.index.year.unique()):
        window = daily[(daily.index >= f"{year}-01-01") & (daily.index <= f"{year}-06-30")]
        if window.empty:
            continue
        hit_a = window.index[window["actual"] >= threshold]
        hit_p = window.index[window["predicted"] >= threshold]
        first_a = hit_a[0] if len(hit_a) else pd.NaT
        first_p = hit_p[0] if len(hit_p) else pd.NaT
        late = (first_p - first_a).days if pd.notna(first_a) and pd.notna(first_p) else None
        records.append({"year": year, "actual": first_a, "predicted": first_p, "late_days": late})
    return curve, pd.DataFrame(records)


def regional_daily(test: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Regional mean actual, predicted and persistence DHW per target date."""
    return (
        test.assign(predicted=predicted, actual=test[TARGET], persistence=test["dhw"])
        .groupby("target_date")[["actual", "predicted", "persistence"]]
        .mean()
        .sort_index()
    )


def fit(
    train: pd.DataFrame, columns: list[str], model_cfg: dict[str, Any]
) -> RandomForestRegressor:
    """Fit the configured Random Forest on ``columns``."""
    params = {k: v for k, v in model_cfg.items() if k != "type"}
    return RandomForestRegressor(**params).fit(train[columns], train[TARGET])


def compute(
    frame: pd.DataFrame, config: dict[str, Any] = CONFIG
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Result rows, the regional daily series, and per-cell test predictions (full model)."""
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float | None, unit: str | None = None, **dims: Any) -> None:
        row: dict[str, Any] = {"metric": metric, "value": value, "unit": unit}
        if dims:
            row["dims"] = dims
        rows.append(row)

    engineered = build_features(frame, config["horizon_days"])
    train, test, embargoed = chronological_split(
        engineered, config["train_years"], config["test_years"]
    )
    add("input_rows", len(frame), "rows")
    add("grid_cells", frame[["latitude", "longitude"]].drop_duplicates().shape[0], "cells")
    add("n_train", len(train), "rows")
    add("n_test", len(test), "rows")
    add("embargoed_rows", embargoed, "rows")
    actual = test[TARGET].to_numpy()
    add("target_mean", float(actual.mean()), "°C-weeks")
    add("target_std", float(actual.std()), "°C-weeks")
    add("rmse_baseline_mean", rmse(actual, np.full_like(actual, train[TARGET].mean())), "°C-weeks")
    persistence = rmse(actual, test["dhw"].to_numpy())
    add("rmse_baseline_persistence", persistence, "°C-weeks")

    t_cfg = config["timing"]
    daily_out = cells_out = None
    for name, spec in config["ablation"].items():
        columns = feature_columns(spec["variables"], spec["diffs"])
        log.info("fitting %s (%d features) on %d rows", name, len(columns), len(train))
        model = fit(train, columns, config["model"])
        predicted = model.predict(test[columns])
        daily = regional_daily(test, predicted)
        curve, crossings = timing(daily, t_cfg["threshold_dhw"], t_cfg["max_shift_days"])
        best = min(curve, key=curve.get)
        late = [d for d in crossings["late_days"] if pd.notna(d)]
        score = rmse(actual, predicted)
        add("ablation_n_features", len(columns), feature_set=name)
        add("ablation_rmse", score, "°C-weeks", feature_set=name)
        add("ablation_r2", float(r2_score(actual, predicted)), feature_set=name)
        add("ablation_skill_vs_persistence", 1 - score / persistence, feature_set=name)
        add("ablation_best_shift", best, "days", feature_set=name)
        add("ablation_mean_late", float(np.mean(late)) if late else None, "days", feature_set=name)
        if name != "full":
            del model
            continue
        add("rmse", score, "°C-weeks")
        add("mae", float(mean_absolute_error(actual, predicted)), "°C-weeks")
        add("r2", float(r2_score(actual, predicted)))
        add("skill_vs_persistence", 1 - score / persistence)
        add("best_shift_days", best, "days")
        add("rmse_at_best_shift", curve[best], "°C-weeks")
        add("rmse_unshifted", curve[0], "°C-weeks")
        add("mean_late_days", float(np.mean(late)) if late else None, "days")
        for shift, value in curve.items():
            add("shift_rmse", value, "°C-weeks", shift=shift)
        for c in crossings.itertuples():
            add(
                "crossing_late_days",
                None if pd.isna(c.late_days) else float(c.late_days),
                "days",
                year=int(c.year),
                actual=None if pd.isna(c.actual) else c.actual.date().isoformat(),
                predicted=None if pd.isna(c.predicted) else c.predicted.date().isoformat(),
            )
        for feature, importance in zip(columns, model.feature_importances_, strict=True):
            add("feature_importance", float(importance), feature=feature)
        daily_out = daily.reset_index()
        cells_out = (
            test[["target_date", "latitude", "longitude", TARGET]]
            .rename(columns={TARGET: "actual"})
            .assign(predicted=predicted.astype("float32"))
        )
        del model
    return rows, daily_out, cells_out


def _parquet(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def load_dataset(client: Client, dataset: dict[str, Any]) -> pd.DataFrame:
    """Download the gzipped CSV, verify the sha256 of its uncompressed bytes, and parse it."""
    bucket, path = dataset["storage_path"].split("/", 1)
    data = gzip.decompress(client.storage.from_(bucket).download(path))
    if sha256_bytes(data) != dataset["sha256"]:
        raise RuntimeError(f"{path} does not match its recorded sha256")
    return pd.read_csv(io.BytesIO(data), parse_dates=["date"])


def run(client: Client, *, allow_dirty: bool = False) -> int | None:
    """Execute the experiment; returns the run id, or ``None`` if skipped."""
    experiment_id = sync_experiments(client)[EXPERIMENT_SLUG]
    rows = (
        client.table("datasets")
        .select("id, sha256, storage_path")
        .eq("slug", DATASET_SLUG)
        .execute()
        .data
    )
    if not rows:
        raise RuntimeError(f"Dataset {DATASET_SLUG} missing; run pipelines.ingest.crw_erddap")
    dataset = rows[0]
    run_id = start_run(
        client,
        experiment_id=experiment_id,
        config=CONFIG,
        git_sha=git_sha(allow_dirty=allow_dirty),
        datasets={dataset["id"]: dataset["sha256"]},
    )
    if run_id is None:
        return None
    with tracked(client, run_id):
        results, daily, cells = compute(load_dataset(client, dataset), CONFIG)
        write_results(client, run_id, results)
        for name, frame in (("daily", daily), ("cells", cells)):
            blob = _parquet(frame)
            storage_path = f"{EXPERIMENT_SLUG}/{run_id}/{name}.parquet"
            upload(client, ARTIFACT_BUCKET, storage_path, blob, "application/vnd.apache.parquet")
            record_artifact(
                client,
                run_id,
                kind="table",
                storage_path=storage_path,
                data=blob,
                content_type="application/vnd.apache.parquet",
            )
    return run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-dirty", action="store_true", help="record a run from uncommitted code"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_id = run(get_client(), allow_dirty=args.allow_dirty)
    log.info(
        "skipped: identical run already succeeded" if run_id is None else f"run {run_id} succeeded"
    )


if __name__ == "__main__":
    main()
