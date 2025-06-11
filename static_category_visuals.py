"""
static_category_visuals.py
--------------------------
Three slide‑ready Plotly helpers:

1. boxplot_by_category(df, category_col, value_col)
   → One box/strip plot per category.

2. monthly_boxplot_grid(df, category_col, value_col)
   → Facet grid: one boxplot panel per month.

3. weekly_mean_lineplot(df, category_col, value_col)
   → One smoothed mean line per category, week‑over‑week.

Each helper shows the figure and returns it (so you can .write_image(...)).
Requires: pandas, numpy, plotly (and kaleido if you export images).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------------
# 1. Single‑colour box / strip plot
# ------------------------------------------------------------------
def boxplot_by_category(df: pd.DataFrame, category_col: str, value_col: str):
    """Static box‑plus‑strip plot of `value_col` split by `category_col`."""
    fig = px.box(
        df, x=category_col, y=value_col,
        points="all", template="simple_white",
        title=f"Distribution of {value_col!r} by {category_col!r}"
    )
    fig.update_traces(marker=dict(opacity=0.5, jitter=0.3))
    fig.update_layout(
        width=900, height=500,
        xaxis_title=category_col,
        yaxis_title=value_col
    )
    fig.show()
    return fig


# ------------------------------------------------------------------
# 2. Monthly facet grid of boxplots
# ------------------------------------------------------------------
def monthly_boxplot_grid(df: pd.DataFrame, category_col: str, value_col: str):
    """Boxplot for each month; facets laid out in a 3‑column grid."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime.")

    data = df.copy()
    data["month"] = data.index.month_name()

    fig = px.box(
        data, x=category_col, y=value_col,
        color=category_col,
        facet_col="month", facet_col_wrap=3,
        template="simple_white",
        title=f"{value_col!r} by {category_col!r} – month‑by‑month"
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False, width=1100, height=600)
    fig.show()
    return fig


# ------------------------------------------------------------------
# 3. Weekly mean line‑plot (smoothed, single‑hue gradient)
# ------------------------------------------------------------------
def weekly_mean_lineplot(df: pd.DataFrame, category_col: str, value_col: str):
    """Weekly mean of `value_col`, one smoothed line per category."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime.")

    data = df.copy()
    data["week"] = data.index.to_period("W")
    pivot = (data.groupby(["week", category_col])[value_col]
                  .mean()
                  .reset_index()
                  .pivot(index="week", columns=category_col, values=value_col))

    n_lines = len(pivot.columns)
    colours = px.colors.sample_colorscale("Blues",
                                          np.linspace(0.15, 0.95, n_lines))

    fig = go.Figure()
    for i, cat in enumerate(pivot.columns):
        fig.add_trace(
            go.Scatter(
                x=pivot.index.to_timestamp(),
                y=pivot[cat],
                mode="lines+markers",
                name=str(cat),
                line=dict(color=colours[i], width=2,
                          shape="spline", smoothing=1.3),
                marker=dict(size=6, color=colours[i])
            )
        )

    fig.update_layout(
        template="simple_white",
        title=f"Weekly mean {value_col!r} – smoothed lines per {category_col}",
        xaxis_title="Week",
        yaxis_title=f"Mean {value_col}",
        width=900, height=500,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    )
    fig.show()
    return fig


# ------------------------------------------------------------------
# Run a quick demo when executed directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    idx = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    n = len(idx)
    np.random.seed(0)
    df_demo = pd.DataFrame({
        "category": np.random.choice(["A", "B", "C"], size=n),
        "value":    np.random.randn(n).cumsum()
    }, index=idx)

    boxplot_by_category(df_demo, "category", "value")
    monthly_boxplot_grid(df_demo, "category", "value")
    weekly_mean_lineplot(df_demo, "category", "value")
