"""
quantile_bin_plots.py
---------------------
Two Plotly helpers that animate quantile-bin means:

1. animated_quantile_bin_mean(...)
2. animated_quantile_bin_mean_with_outliers(...)

See docstrings in each function for details.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------------------------------------
def _build_frames(df2, x_col, y_col, period, n_bins, exclude_outliers):
    periods = sorted(df2["period"].unique())
    frames = []

    for per in periods:
        sub = df2[df2["period"] == per][[x_col, y_col]].dropna()

        out_mask = np.zeros(len(sub), dtype=bool)
        if exclude_outliers and len(sub):
            q1, q3 = sub[y_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_mask = (sub[y_col] < low) | (sub[y_col] > high)

        if len(sub) < 2:
            continue

        edges = np.quantile(sub[x_col], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            continue

        bin_idx = np.searchsorted(edges, sub[x_col], side="right") - 1
        bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

        mean_x, mean_y = [], []
        for i in range(len(edges) - 1):
            mask = bin_idx == i
            if exclude_outliers:
                mask = mask & (~out_mask)
            if mask.any():
                mean_x.append(sub.loc[mask, x_col].mean())
                mean_y.append(sub.loc[mask, y_col].mean())
            else:
                mean_x.append(np.nan)
                mean_y.append(np.nan)

        line = go.Scatter(
            x=mean_x, y=mean_y,
            mode="lines+markers",
            line=dict(color="steelblue", width=2, shape="spline", smoothing=1.2),
            marker=dict(size=6, color="steelblue"),
            name="Mean"
        )
        traces = [line]

        if exclude_outliers and out_mask.any():
            traces.append(
                go.Scatter(
                    x=sub.loc[out_mask, x_col],
                    y=sub.loc[out_mask, y_col],
                    mode="markers",
                    marker=dict(size=5, color="indianred", opacity=0.55, symbol="circle-open"),
                    name="Outliers"
                )
            )
        frames.append(go.Frame(data=traces, name=per))
    return frames, periods

# ----------------------------------------------------------------------
def _animate_quant_bins(df, x_col, y_col, period, n_bins, excl=False, title_suffix=""):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")
    for c in (x_col, y_col):
        if c not in df.columns:
            raise ValueError(f"{c} missing")

    df2 = df.copy()
    df2["period"] = df2.index.to_series().dt.to_period(period).astype(str)

    frames, periods = _build_frames(df2, x_col, y_col, period, n_bins, excl)
    if not frames:
        raise ValueError("No frame data.")

    fig = go.Figure(
        data=frames[0].data,
        layout=dict(
            title=f"{title_suffix} mean {y_col} in {n_bins} quantile bins of {x_col}",
            xaxis=dict(title=x_col, autorange=True),
            yaxis=dict(title=f"Mean {y_col}", autorange=True),
            template="simple_white",
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=500), fromcurrent=True)]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")])
                ],
                showactive=False, x=0.05, y=1.15
            )],
            sliders=[dict(
                active=0,
                steps=[dict(method="animate", label=p,
                            args=[[p], dict(frame=dict(duration=500), mode="immediate")])
                       for p in periods],
                currentvalue=dict(prefix="Period: ")
            )]
        ),
        frames=frames
    )
    fig.show()
    return fig

# Public helpers -------------------------------------------------------
def animated_quantile_bin_mean(df, x_col, y_col, period="W", n_bins=10):
    """Animated quantile bin means (no outlier handling)."""
    return _animate_quant_bins(df, x_col, y_col, period, n_bins, False, "")

def animated_quantile_bin_mean_with_outliers(df, x_col, y_col, period="W", n_bins=10):
    """Animated quantile bin means with outliers removed + shown in red."""
    return _animate_quant_bins(df, x_col, y_col, period, n_bins, True, "Outlier‑adjusted")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = pd.date_range("2025-01-01", "2025-03-31", freq="H")
    np.random.seed(2)
    demo = pd.DataFrame({
        "x": np.random.randn(len(rng)).cumsum(),
        "y": np.random.randn(len(rng))*10 + 50
    }, index=rng)

    animated_quantile_bin_mean(demo, "x", "y", "W", 10)
    animated_quantile_bin_mean_with_outliers(demo, "x", "y", "W", 10)
