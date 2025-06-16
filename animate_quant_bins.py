
"""
Animated quantile‑bin plot with ±1 standard‑error bars and a zero baseline.

Exposes a single function
    animate_quant_bins(df, x_col, y_col, lq, uq, period, n_bins, excl=False,
                       title_suffix="")

Usage example
-------------
>>> from animate_quant_bins import animate_quant_bins
>>> animate_quant_bins(df, "feature", "target",
...                    lq=0.01, uq=0.99,
...                    period="M", n_bins=10,
...                    title_suffix="My data – ")

Requires: pandas, numpy, plotly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def animate_quant_bins(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lq: float,
    uq: float,
    period: str,
    n_bins: int,
    *,
    excl: bool = False,
    title_suffix: str = "",
) -> None:
    """Create an animated quantile‑bin plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Index must be datetime‑like; it is collapsed to *period*.
    x_col, y_col : str
        Column names of the feature (x) and the response (y).
    lq, uq : float
        Lower / upper quantiles for winsorisation (kept for compatibility).
    period : str
        Pandas offset alias: "M", "W", "D", etc. Frames are built per period.
    n_bins : int
        Number of *quantile* bins along ``x_col``.
    excl : bool, default False
        Reserved for compatibility – currently unused.
    title_suffix : str, default ""
        Prepended to the figure title.
    """

    # ── 1. Prepare data
    df2 = df.copy()
    df2["period"] = df2.index.to_series().dt.to_period(period).astype(str)

    frames = []
    ymins, ymaxs = [], []

    # ── 2. Build frames
    for p, df_p in df2.groupby("period"):
        # quantile bins
        df_p["q"] = pd.qcut(
            df_p[x_col],
            q=n_bins,
            labels=range(n_bins),
            duplicates="drop",
        )

        # stats per bin
        stats = (
            df_p.groupby("q")[[x_col, y_col]]
            .agg({x_col: "mean", y_col: ["mean", "std", "count"]})
        )
        stats.columns = ["x_ctr", "y_mean", "y_std", "n"]
        stats["y_se"] = stats["y_std"] / np.sqrt(stats["n"])

        # record y‑limits
        ymins.append((stats["y_mean"] - stats["y_se"]).min())
        ymaxs.append((stats["y_mean"] + stats["y_se"]).max())

        # piece‑wise line with error bars
        trace_bins = go.Scatter(
            x=stats["x_ctr"],
            y=stats["y_mean"],
            mode="lines+markers",
            name="Bins",
            line=dict(shape="linear", width=2, color="#1f77b4"),
            marker=dict(size=6, color="#1f77b4"),
            error_y=dict(
                type="data",
                array=stats["y_se"],
                visible=True,
                thickness=1.2,
            ),
        )

        frames.append(go.Frame(data=[trace_bins], name=str(p)))

    # ── 3. Fixed global y‑range
    pad = 0.05
    ymin, ymax = min(ymins), max(ymaxs)
    yrange = [ymin - abs(ymin) * pad, ymax + abs(ymax) * pad]

    # ── 4. Figure
    fig = go.Figure(
        data=frames[0].data,
        layout=dict(
            title=f"{title_suffix}mean {y_col} in {n_bins} quantile bins of {x_col}",
            xaxis=dict(title=x_col, autorange=True),
            yaxis=dict(title=f"Mean {y_col}", range=yrange),
            template="simple_white",
            # zero baseline
            shapes=[dict(
                type="line",
                xref="paper", yref="y",
                x0=0, x1=1, y0=0, y1=0,
                line=dict(color="black", width=1),
            )],
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.02, y=1.15,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=500), fromcurrent=True)]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Period: "),
                steps=[
                    dict(method="animate", label=f.name,
                         args=[[f.name], dict(frame=dict(duration=500), mode="immediate")])
                    for f in frames
                ],
            )],
        ),
        frames=frames,
    )

    fig.show()
