"""
category_mean_month_lines.py
----------------------------
Function: category_mean_by_month_lineplot(df, category_col, value_col)

Creates a static line chart:
    • X‑axis  : categories (categorical variable)
    • Y‑axis  : mean(value_col) for each category
    • One line per calendar month (January … December)

Colour palette is a single‑hue gradient (Blues) for a clean slide‑ready look.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def category_mean_by_month_lineplot(df: pd.DataFrame,
                                    category_col: str,
                                    value_col: str):
    """
    Parameters
    ----------
    df : DataFrame with DatetimeIndex
    category_col : str
        Categorical column on x‑axis
    value_col : str
        Numeric column to average
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")
    for col in (category_col, value_col):
        if col not in df.columns:
            raise ValueError(f"{col!r} not in DataFrame.")

    # Add month label
    data = df.copy()
    data["month"] = data.index.month_name()

    # Mean per (month, category)
    means = (data
             .groupby(["month", category_col])[value_col]
             .mean()
             .reset_index())

    # Axis ordering
    cat_order   = sorted(data[category_col].dropna().unique())
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    months_present = [m for m in month_order if m in means["month"].unique()]

    # Single-hue gradient colours
    n = len(months_present)
    colours = px.colors.sample_colorscale("Blues",
                                          np.linspace(0.15, 0.9, n))

    fig = go.Figure()
    for i, m in enumerate(months_present):
        sub = means[means["month"] == m]
        y_vals = [sub[sub[category_col] == c][value_col].iloc[0]
                  if c in sub[category_col].values else None
                  for c in cat_order]
        fig.add_trace(
            go.Scatter(
                x=cat_order,
                y=y_vals,
                mode="lines+markers",
                name=m,
                line=dict(color=colours[i], width=2),
                marker=dict(size=6, color=colours[i])
            )
        )

    fig.update_layout(
        template="simple_white",
        title=f"Mean {value_col!r} by {category_col!r} – one line per month",
        xaxis_title=category_col,
        yaxis_title=f"Mean {value_col}",
        width=900, height=500,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    )
    fig.show()
    return fig

# Demo when executed directly
if __name__ == "__main__":
    rng = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    np.random.seed(1)
    demo = pd.DataFrame({
        "category": np.random.choice(["A", "B", "C"], size=len(rng)),
        "value":    np.random.randn(len(rng)).cumsum()
    }, index=rng)

    category_mean_by_month_lineplot(demo, "category", "value")
