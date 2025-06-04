
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, VBox, Output
from IPython.display import display
from datetime import timedelta

def hourly_mean_slider_plot(df, feature_col='feature', window_days=7):
    """
    Interactive hourly mean plot using a slider to step through time in fixed increments.
    """
    # Prepare data
    df = df.copy()
    df = df.sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    if 'hour' not in df.columns:
        df['hour'] = df.index.hour

    # Calculate time windows
    start_date = df.index.min().normalize()
    end_date = df.index.max().normalize()
    total_days = (end_date - start_date).days
    num_steps = max(1, total_days // window_days)

    out = Output()

    def plot(step):
        out.clear_output(wait=True)
        with out:
            s = start_date + timedelta(days=step * window_days)
            e = s + timedelta(days=window_days)
            window_df = df[(df.index >= s) & (df.index < e)]

            if window_df.empty:
                print(f"No data from {s.date()} to {e.date()}")
                return

            hourly_means = window_df.groupby('hour')[feature_col].mean()

            fig, ax = plt.subplots(figsize=(10, 5))
            hourly_means.plot(kind='bar', ax=ax)
            ax.set_title(f"{feature_col} Mean by Hour ({s.date()} to {e.date()})")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Mean Value")
            ax.grid(True)
            plt.tight_layout()
            plt.show()

    slider = IntSlider(value=0, min=0, max=num_steps - 1, step=1, description='Window')
    slider.observe(lambda change: plot(change['new']), names='value')

    display(VBox([slider, out]))
    plot(0)
