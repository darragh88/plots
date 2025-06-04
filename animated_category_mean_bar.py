import pandas as pd
import numpy as np
import plotly.graph_objects as go

def animated_category_mean_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    period: str = "W"
):
    """
    Creates an animated bar chart of mean(value_col) for each category in category_col,
    with a slider over time periods (day/week/month).

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and contain columns category_col and value_col.
    category_col : str
        Name of the categorical column.
    value_col : str
        Name of the numerical target column whose mean will be plotted per category.
    period : str, default "W"
        Time period for grouping. One of:
          "D" = daily periods,
          "W" = weekly periods,
          "M" = monthly periods.
    """
    # 1) VALIDATE INPUTS
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("`df` must have a DatetimeIndex.")
    for c in (category_col, value_col):
        if c not in df.columns:
            raise ValueError(f"Column {c!r} not found in DataFrame.")
    if period not in ("D", "W", "M"):
        raise ValueError("`period` must be one of 'D', 'W', or 'M'.")

    # 2) ADD "period" COLUMN
    df2 = df.copy()
    df2["period"] = df2.index.to_series().dt.to_period(period).astype(str)
    unique_periods = sorted(df2["period"].unique())

    # 3) COLLECT GLOBAL CATEGORY LIST
    all_categories = sorted(df2[category_col].dropna().unique())

    # 4) BUILD FRAMES: compute mean(value_col) per category for each period
    frames = []
    for per in unique_periods:
        df_period = df2[df2["period"] == per]
        # Compute per-category means
        means = df_period.groupby(category_col)[value_col].mean()
        # Ensure every category in all_categories appears
        y_vals = [float(means.get(cat, np.nan)) for cat in all_categories]
        bar_trace = go.Bar(
            x=all_categories,
            y=y_vals,
            marker=dict(color="steelblue"),
            name=f"Mean of {value_col}",
            showlegend=False
        )
        frames.append(go.Frame(data=[bar_trace], name=per))

    # 5) INITIAL DATA (first period)
    first_per = unique_periods[0]
    df_first = df2[df2["period"] == first_per]
    means_first = df_first.groupby(category_col)[value_col].mean()
    init_y = [float(means_first.get(cat, np.nan)) for cat in all_categories]
    init_bar = go.Bar(
        x=all_categories,
        y=init_y,
        marker=dict(color="steelblue"),
        name=f"Mean of {value_col}",
        showlegend=False
    )

    # 6) SLIDER STEPS
    slider_steps = []
    for per in unique_periods:
        step = dict(
            method="animate",
            label=per,
            args=[
                [per],
                {"frame": {"duration": 500, "redraw": False}, "mode": "immediate"}
            ]
        )
        slider_steps.append(step)

    sliders = [{
        "active": 0,
        "pad": {"t": 60},
        "steps": slider_steps,
        "currentvalue": {"prefix": "Period: ", "font": {"size": 16}}
    }]

    # 7) PLAY/PAUSE BUTTONS
    play_pause_buttons = [
        {
            "label": "Play",
            "method": "animate",
            "args": [
                None,
                {"frame": {"duration": 500, "redraw": False}, "fromcurrent": True}
            ]
        },
        {
            "label": "Pause",
            "method": "animate",
            "args": [
                [None],
                {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}
            ]
        }
    ]

    # 8) LAYOUT with autorange so axes adjust each frame
    layout = go.Layout(
        title=f"Mean of {value_col!r} by {category_col!r} — animated by {period}",
        xaxis=dict(title=category_col, autorange=True),
        yaxis=dict(title=f"Mean of {value_col}", autorange=True),
        updatemenus=[{
            "type": "buttons",
            "buttons": play_pause_buttons,
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "x": 0.1,
            "y": 1.15,
            "xanchor": "right"
        }],
        sliders=sliders,
        bargap=0.2
    )

    # 9) BUILD & SHOW FIGURE
    fig = go.Figure(data=[init_bar], layout=layout, frames=frames)
    fig.show()

# ------------------------------
# EXAMPLE USAGE
# ------------------------------
if __name__ == "__main__":
    # Toy DataFrame: daily index Jan 1–Jan 30, 2025
    idx = pd.date_range("2025-01-01", "2025-01-30", freq="D")
    n = len(idx)
    np.random.seed(0)
    df_toy = pd.DataFrame({
        "category": np.random.choice(["A", "B", "C"], size=n),
        "value":    np.random.randn(n).cumsum() + 10
    }, index=idx)

    # Animate mean(value) per category, grouped by week ("W") with 3 categories
    animated_category_mean_bar(
        df=df_toy,
        category_col="category",
        value_col="value",
        period="W"
    )
