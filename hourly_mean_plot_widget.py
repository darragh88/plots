
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import Button, HBox, VBox, Output
from datetime import timedelta

def hourly_mean_stepping_plot(df, feature_col='feature', hour_col='hour', window_days=7):
    """
    Interactive plot that steps through time in fixed increments.
    """
    # Ensure datetime index is sorted
    df = df.sort_index()
    df[hour_col] = df.index.hour

    # State
    state = {
        "start": df.index.min(),
        "window": timedelta(days=window_days),
        "df": df
    }

    out = Output()

    def plot():
        with out:
            out.clear_output()
            start = state["start"]
            end = start + state["window"]
            mask = (df.index >= start) & (df.index < end)
            filtered_df = df.loc[mask]
            if filtered_df.empty:
                print(f"No data from {start.date()} to {end.date()}")
                return
            hourly_mean = filtered_df.groupby(hour_col)[feature_col].mean()
            plt.figure(figsize=(10, 5))
            hourly_mean.plot(kind='bar')
            plt.xlabel('Hour')
            plt.ylabel(f'Mean of {feature_col}')
            plt.title(f'Hourly Mean from {start.date()} to {end.date()}')
            plt.grid(True)
            plt.show()

    def on_next_clicked(b):
        state["start"] += state["window"]
        plot()

    def on_prev_clicked(b):
        state["start"] -= state["window"]
        plot()

    # Buttons
    btn_next = Button(description="Next")
    btn_prev = Button(description="Previous")
    btn_next.on_click(on_next_clicked)
    btn_prev.on_click(on_prev_clicked)

    # Initial plot
    plot()
    display(VBox([HBox([btn_prev, btn_next]), out]))
