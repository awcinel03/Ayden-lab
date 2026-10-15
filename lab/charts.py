"""Shared Altair chart builders."""

from __future__ import annotations

import altair as alt
import pandas as pd


def loglog_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None = None,
    x_title: str,
    y_title: str,
) -> alt.Chart:
    """Scatter plot on log-log axes. Rows with non-positive ``x`` or ``y`` are dropped."""
    data = df[(df[x] > 0) & (df[y] > 0)]
    tooltip = [alt.Tooltip(x, title=x_title), alt.Tooltip(y, title=y_title)]
    encoding = {
        "x": alt.X(x, title=x_title, scale=alt.Scale(type="log")),
        "y": alt.Y(y, title=y_title, scale=alt.Scale(type="log")),
        "tooltip": tooltip,
    }
    if color:
        encoding["color"] = alt.Color(color, title=color.capitalize())
        tooltip.insert(0, alt.Tooltip(color, title=color.capitalize()))
    return alt.Chart(data).mark_circle(size=40, opacity=0.7).encode(**encoding).interactive()
