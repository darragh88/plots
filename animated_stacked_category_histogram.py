
"""
animated_stacked_category_histogram.py
--------------------------------------
A reusable Plotly helper that animates stacked‑bar histograms over time.

Function
--------
animated_stacked_category_histogram(df, x_category, stack_category, period="W")

  • df              : pandas DataFrame **with a DatetimeIndex**.
  • x_category      : column whose unique values form the x‑axis categories.
                      Special case: the literal string "hour" derives
                      00‑23 from the index when no such column exists.
  • stack_category  : column whose values colour‑stack the bars.
  • period          : "D"=daily, "W"=weekly, "M"=monthly, "Y"=yearly.

Returns a Plotly Figure (already shown).

Example
-------
>>> import pandas as pd, numpy as np
>>> from animated_stacked_category_histogram import animated_stacked_category_histogram
>>> idx = pd.date_range("2025-01-01", "2025-03-31 23:00", freq="H")
>>> rng = np.random.default_rng(0)
>>> df_demo = pd.DataFrame({
...     "hour_bucket": idx.hour.astype(str).str.zfill(2),
...     "device": rng.choice(["mobile", "desktop", "tablet"], size=len(idx)),
... }, index=idx)
>>> animated_stacked_category_histogram(df_demo, "hour_bucket", "device", period="W")
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative


def animated_stacked_category_histogram(
    df: pd.DataFrame,
    x_category: str,
    stack_category: str,
    period: str = "W"
) -> go.Figure:
    """Animated stacked‑bar chart of categorical counts through time intervals."""

    # ------------------------------------------------------------------ #
    # 1) VALIDATION & PREP
    # ------------------------------------------------------------------ #
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("`df` must have a DatetimeIndex.")

    df2 = df.copy()

    # Derive an "hour" column on demand
    if x_category not in df2.columns:
        if x_category.lower() == "hour":
            df2["hour"] = df2.index.hour.astype(str).str.zfill(2)
        else:
            raise ValueError(f"Column {x_category!r} not found in DataFrame.")

    for col in (x_category, stack_category):
        if col not in df2.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")

    if period not in ("D", "W", "M", "Y"):
        raise ValueError("`period` must be one of 'D', 'W', 'M', or 'Y'.")

    # Friendly label for the title
    period_names = {"D": "day", "W": "week", "M": "month", "Y": "year"}

    # ------------------------------------------------------------------ #
    # 2) ADD PERIOD LABEL
    # ------------------------------------------------------------------ #
    df2["interval"] = df2.index.to_period(period).astype(str)
    unique_intervals = sorted(df2["interval"].unique())

    # ------------------------------------------------------------------ #
    # 3) GLOBAL CATEGORY ORDER
    # ------------------------------------------------------------------ #
    x_vals = sorted(df2[x_category].dropna().unique())
    stack_vals = sorted(df2[stack_category].dropna().unique())

    palette = qualitative.Plotly
    colors = {s: palette[i % len(palette)] for i, s in enumerate(stack_vals)}

    # ------------------------------------------------------------------ #
    # 4) BUILD FRAMES
    # ------------------------------------------------------------------ #
    frames = []
    for intv in unique_intervals:
        counts = (
            df2[df2["interval"] == intv]
            .groupby([x_category, stack_category])
            .size()
            .unstack(fill_value=0)
            .reindex(index=x_vals, columns=stack_vals, fill_value=0)
        )

        traces = [
            go.Bar(
                x=x_vals,
                y=counts[s].tolist(),
                name=str(s),
                marker_color=colors[s],
                hovertemplate=f"{x_category}: %{{x}}<br>{stack_category}: {s}<br>count: %{{y}}<extra></extra>",
                showlegend=False
            )
            for s in stack_vals
        ]
        frames.append(go.Frame(data=traces, name=intv))

    # ------------------------------------------------------------------ #
    # 5) INITIAL TRACES
    # ------------------------------------------------------------------ #
    first_counts = (
        df2[df2["interval"] == unique_intervals[0]]
        .groupby([x_category, stack_category])
        .size()
        .unstack(fill_value=0)
        .reindex(index=x_vals, columns=stack_vals, fill_value=0)
    )
    first_traces = [
        go.Bar(
            x=x_vals,
            y=first_counts[s].tolist(),
            name=str(s),
            marker_color=colors[s],
            hovertemplate=f"{x_category}: %{{x}}<br>{stack_category}: {s}<br>count: %{{y}}<extra></extra>",
            showlegend=True
        )
        for s in stack_vals
    ]

    # ------------------------------------------------------------------ #
    # 6) SLIDER & BUTTONS
    # ------------------------------------------------------------------ #
    slider_steps = [
        dict(
            method="animate",
            label=intv,
            args=[[intv], {"frame": {"duration": 500, "redraw": False}, "mode": "immediate"}],
        )
        for intv in unique_intervals
    ]

    sliders = [dict(
        active=0,
        pad={"t": 60},
        steps=slider_steps,
        currentvalue={"prefix": "Interval: ", "font": {"size": 16}},
    )]

    play_pause = [
        dict(
            label="Play",
            method="animate",
            args=[None, {"frame": {"duration": 500, "redraw": False}, "fromcurrent": True}],
        ),
        dict(
            label="Pause",
            method="animate",
            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
        ),
    ]

    # ------------------------------------------------------------------ #
    # 7) LAYOUT
    # ------------------------------------------------------------------ #
    layout = go.Layout(
        title=f"Counts by {x_category!r} (stacked by {stack_category!r}) — one {period_names[period]} per frame",
        xaxis=dict(title=x_category, type="category"),
        yaxis=dict(title="Count", autorange=True),
        barmode="stack",
        bargap=0.15,
        updatemenus=[dict(
            type="buttons",
            buttons=play_pause,
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=False,
            x=0.1,
            y=1.15,
            xanchor="right",
        )],
        sliders=sliders,
        legend_title=stack_category,
    )

    # ------------------------------------------------------------------ #
    # 8) FIGURE
    # ------------------------------------------------------------------ #
    fig = go.Figure(data=first_traces, layout=layout, frames=frames)
    fig.show()
    return fig


if __name__ == "__main__":
    # Quick demo if executed directly
    idx = pd.date_range("2025-01-01", "2025-03-31 23:00", freq="H")
    rng = np.random.default_rng(0)
    df_demo = pd.DataFrame({
        "hour_bucket": idx.hour.astype(str).str.zfill(2),
        "device": rng.choice(["mobile", "desktop", "tablet"], size=len(idx)),
    }, index=idx)
    animated_stacked_category_histogram(
        df_demo,
        x_category="hour_bucket",
        stack_category="device",
        period="W"
    )
