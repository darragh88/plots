
"""
Animated quantile–bin comparison plot (Actual vs. Predicted)
============================================================

See docstring of animate_quant_bins_dual for details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _bin_stats(df_p: pd.DataFrame, x_col: str, y_col: str, n_bins: int) -> pd.DataFrame:
    """Return centre‑x, mean‑y, std, count and se for each quantile bin."""
    work = df_p[[x_col, y_col]].copy()
    work["q"] = pd.qcut(work[x_col], q=n_bins,
                         labels=range(n_bins), duplicates="drop")

    g = (
        work.groupby("q")[[x_col, y_col]]
            .agg({x_col: "mean", y_col: ["mean", "std", "count"]})
    )
    g.columns = ["x_ctr", "y_mean", "y_std", "n"]
    g["y_se"] = g["y_std"] / np.sqrt(g["n"])
    return g


def animate_quant_bins_dual(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_true_col: str,
    y_pred_col: str,
    lq: float,
    uq: float,
    period: str,
    n_bins: int,
    title_suffix: str = "",
    show_outliers: bool = True,
) -> None:
    """Animated Actual vs. Predicted quantile‑bin curves."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df2 = df.copy()
    df2["period"] = df2.index.to_series().dt.to_period(period).astype(str)

    frames = []
    ymins, ymaxs = [], []

    # Colours
    col_true = "#1f77b4"   # blue
    col_pred = "#ff7f0e"   # orange

    for p, df_p in df2.groupby("period"):
        stats_true = _bin_stats(df_p, x_col, y_true_col, n_bins)
        stats_pred = _bin_stats(df_p, x_col, y_pred_col, n_bins)

        ymins.append(min((stats_true["y_mean"] - stats_true["y_se"]).min(),
                         (stats_pred["y_mean"] - stats_pred["y_se"]).min()))
        ymaxs.append(max((stats_true["y_mean"] + stats_true["y_se"]).max(),
                         (stats_pred["y_mean"] + stats_pred["y_se"]).max()))

        # True curve
        trace_true = go.Scatter(
            x=stats_true["x_ctr"],
            y=stats_true["y_mean"],
            mode="lines+markers",
            name="Actual",
            line=dict(shape="linear", width=2, color=col_true),
            marker=dict(size=6, color=col_true),
            error_y=dict(type="data", array=stats_true["y_se"],
                         visible=True, thickness=1.2)
        )

        # Predicted curve
        trace_pred = go.Scatter(
            x=stats_pred["x_ctr"],
            y=stats_pred["y_mean"],
            mode="lines+markers",
            name="Predicted",
            line=dict(shape="linear", width=2, dash="dash", color=col_pred),
            marker=dict(size=6, color=col_pred),
            error_y=dict(type="data", array=stats_pred["y_se"],
                         visible=True, thickness=1.2)
        )

        traces = [trace_true, trace_pred]

        # Outliers (actuals only)
        if show_outliers:
            lo, hi = df_p[y_true_col].quantile([lq, uq])
            mask_out = (df_p[y_true_col] < lo) | (df_p[y_true_col] > hi)
            if mask_out.any():
                trace_out = go.Scatter(
                    x=df_p.loc[mask_out, x_col],
                    y=df_p.loc[mask_out, y_true_col],
                    mode="markers",
                    name="Outliers",
                    marker=dict(size=5, symbol="circle-open", opacity=0.35,
                                color="#d62728"),
                    hoverinfo="skip",
                )
                traces.append(trace_out)

        frames.append(go.Frame(data=traces, name=str(p)))

    pad = 0.05
    yrange = [min(ymins) - abs(min(ymins)) * pad,
              max(ymaxs) + abs(max(ymaxs)) * pad]

    fig = go.Figure(
        data=frames[0].data,
        layout=dict(
            title=(f"{title_suffix}mean {y_true_col} & {y_pred_col} in {n_bins} "
"
                   f"quantile bins of {x_col}"),
            xaxis=dict(title=x_col, autorange=True),
            yaxis=dict(title="Mean value", range=yrange),
            template="simple_white",
            shapes=[dict(
                type="line", xref="paper", yref="y",
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
