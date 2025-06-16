
"""
Animated quantile-bin plot with ±1 standard‑error bars, a zero baseline,
and optional outlier markers that never influence the y‑axis scale.

Function
--------
animate_quant_bins(df, x_col, y_col, lq, uq, period, n_bins,
                   *, excl=False, title_suffix="", show_outliers=True)

If *show_outliers* is True (default) the function plots all observations
outside the [lq, uq] quantile band *per period* as hollow red markers.
The y‑axis limits are computed **only** from the bin‑mean ±1 SE envelope,
so extreme points never squash the interesting part of the chart.

Dependencies: pandas, numpy, plotly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _bin_stats(df_p: pd.DataFrame, x_col: str, y_col: str, n_bins: int) -> pd.DataFrame:
    """Return centre‑x, mean‑y and ±1 SE per quantile bin."""
    # allocate to quantile bins (labels = 0 .. n_bins‑1)
    df_p = df_p.copy()
    df_p["q"] = pd.qcut(df_p[x_col], q=n_bins,
                         labels=range(n_bins), duplicates="drop")

    stats = (
        df_p.groupby("q")[[x_col, y_col]]
        .agg({x_col: "mean", y_col: ["mean", "std", "count"]})
    )
    stats.columns = ["x_ctr", "y_mean", "y_std", "n"]
    stats["y_se"] = stats["y_std"] / np.sqrt(stats["n"])

    return stats


def animate_quant_bins(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lq: float,
    uq: float,
    period: str,
    n_bins: int,
    *,
    excl: bool = False,       # kept for backward compatibility
    title_suffix: str = "",
    show_outliers: bool = True,
) -> None:
    """Create an animated quantile‑bin plot.

    Parameters
    ----------
    df : DataFrame
        Must have a DatetimeIndex.
    x_col, y_col : str
        Columns for x and y.
    lq, uq : float
        Lower / upper quantiles used to flag outliers (per *period*).
    period : str
        Pandas offset alias ("M", "W", "D", etc.).
    n_bins : int
        Number of quantile bins.
    excl : bool, default False
        Reserved.
    title_suffix : str
        Prepended to the figure title.
    show_outliers : bool, default True
        Whether to add the outlier scatter trace.
    """

    # ────────────────── 1. Prepare data ───────────────────────────────────────
    df2 = df.copy()
    if not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df2["period"] = df2.index.to_series().dt.to_period(period).astype(str)

    frames = []
    ymins, ymaxs = [], []

    # ────────────────── 2. Build frames ───────────────────────────────────────
    for p, df_p in df2.groupby("period"):
        stats = _bin_stats(df_p, x_col, y_col, n_bins)

        # Collect y‑extremes from bin curve (NOT from outliers)
        ymins.append((stats["y_mean"] - stats["y_se"]).min())
        ymaxs.append((stats["y_mean"] + stats["y_se"]).max())

        # Bin curve trace
        trace_bins = go.Scatter(
            x=stats["x_ctr"],
            y=stats["y_mean"],
            mode="lines+markers",
            name="Bins",
            line=dict(shape="linear", width=2, color="#1f77b4"),
            marker=dict(size=6, color="#1f77b4"),
            error_y=dict(type="data", array=stats["y_se"],
                         visible=True, thickness=1.2),
        )

        traces = [trace_bins]

        # ─── Outliers (optional) ──────────────────────────────────────────────
        if show_outliers:
            lo, hi = df_p[y_col].quantile([lq, uq])
            mask_out = (df_p[y_col] < lo) | (df_p[y_col] > hi)

            if mask_out.any():
                trace_out = go.Scatter(
                    x=df_p.loc[mask_out, x_col],
                    y=df_p.loc[mask_out, y_col],
                    mode="markers",
                    name="Outliers",
                    marker=dict(size=5, symbol="circle-open", opacity=0.4,
                                color="#d62728"),
                    hoverinfo="skip",
                )
                traces.append(trace_out)

        # Assemble frame
        frames.append(go.Frame(data=traces, name=str(p)))

    # ────────────────── 3. Fixed y‑axis range ─────────────────────────────────
    pad = 0.05
    ymin, ymax = min(ymins), max(ymaxs)
    yrange = [ymin - abs(ymin) * pad, ymax + abs(ymax) * pad]

    # ────────────────── 4. Figure & controls ─────────────────────────────────
    fig = go.Figure(
        data=frames[0].data,
        layout=dict(
            title=f"{title_suffix}mean {y_col} in {n_bins} quantile bins of {x_col}",
            xaxis=dict(title=x_col, autorange=True),
            yaxis=dict(title=f"Mean {y_col}", range=yrange),
            template="simple_white",
            # zero baseline (full width)
            shapes=[dict(
                type="line", xref="paper", yref="y",
                x0=0, x1=1, y0=0, y1=0,
                line=dict(color="black", width=1),
            )],
            # Play / Pause buttons
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.02, y=1.15,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=500),
                                          fromcurrent=True)]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0),
                                            mode="immediate")]),
                ],
            )],
            # Period slider
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Period: "),
                steps=[
                    dict(method="animate", label=f.name,
                         args=[[f.name], dict(frame=dict(duration=500),
                                              mode="immediate")])
                    for f in frames
                ],
            )],
        ),
        frames=frames,
    )

    fig.show()
