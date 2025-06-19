
"""
visual_analytics_japan.py
=========================
Three visual analytics helpers for the Japanese imbalance‑price dataset,
requiring only pandas, numpy, matplotlib, seaborn, and networkx
(no GNN libraries).

1. tie_line_heatmap(G, price_df, out)
   • Shows a heat‑map of daily maximum price spread relative to each
     tie-line’s capacity.

2. animate_price_graph(G, target_df, feature_is_numeric, out)
   • Creates an MP4 (or GIF) animation of the 10‑node grid.  Nodes are
     coloured by either:
       – a numeric target (e.g. imbalance price), OR
       – a boolean / categorical flag (set feature_is_numeric=False).

3. rolling_mae_plot(price_df, window, out)
   • Plots the rolling Mean Absolute Error of the simplest persistence
     baseline (ŷ_t = y_{t-1}) for all regions.

Usage
-----
• `python visual_analytics_japan.py --demo`
    runs with synthetic data and writes results into ./outputs/

• Replace the dummy‑data function with your own data loading logic, or
  import these functions into a notebook:

    from visual_analytics_japan import (
        tie_line_heatmap, animate_price_graph, rolling_mae_plot
    )

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.animation as animation

# ------------------------------------------------------------------ #
# 1 · Heat‑map of congestion proxy                                    #
# ------------------------------------------------------------------ #
def tie_line_heatmap(G: nx.Graph,
                     price_df: pd.DataFrame,
                     out: Path):
    """Create a heat‑map with rows = tie‑lines and columns = days.

    G         : networkx.Graph with edge attr 'capacity'
    price_df  : DataFrame (datetime index, columns = regions) numeric prices
    out       : Path to .png file
    """
    daily_spread = price_df.resample('D').apply(lambda x: x.max() - x.min())

    edge_labels = [f"{u}-{v}" for u, v in G.edges()]
    util = pd.DataFrame(index=daily_spread.index,
                        columns=edge_labels,
                        dtype=float)

    for (u, v) in G.edges():
        cap = G[u][v]['capacity']
        col = f"{u}-{v}"
        util[col] = daily_spread.max(axis=1) / cap

    plt.figure(figsize=(len(edge_labels) * 0.6 + 2, 6))
    sns.heatmap(util.T,
                cmap='YlOrRd',
                cbar_kws={'label': 'Spread / Capacity'})
    plt.title('Daily Price Divergence relative to Tie‑line Capacity')
    plt.ylabel('Tie‑line')
    plt.xlabel('Day')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[✓] Heat‑map saved → {out}")

# ------------------------------------------------------------------ #
# 2 · Animated graph                                                 #
# ------------------------------------------------------------------ #
def animate_price_graph(G: nx.Graph,
                        target_df: pd.DataFrame,
                        feature_is_numeric: bool,
                        out: Path,
                        fps: int = 5):
    """Animate the grid coloured by target values through time.

    target_df must share columns with G.nodes (same ordering not needed).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=2)

    if feature_is_numeric:
        vmin = target_df.min().min()
        vmax = target_df.max().max()
        cmap = plt.cm.plasma
    else:
        cmap = {True: 'tomato', False: 'skyblue'}

    def update(i):
        ax.clear()
        ts = target_df.index[i]
        series = target_df.loc[ts]
        if feature_is_numeric:
            colors = [
                cmap((series[n] - vmin) / (vmax - vmin + 1e-9))
                for n in G.nodes
            ]
        else:
            colors = [cmap[bool(series[n])] for n in G.nodes]
        nx.draw_networkx(G, pos,
                         node_color=colors,
                         edge_color='gray',
                         with_labels=True,
                         ax=ax)
        ax.set_title(ts.strftime('%Y-%m-%d %H:%M'))
        ax.axis('off')

    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=len(target_df),
                                  interval=200)
    ani.save(out, fps=fps)
    plt.close(fig)
    print(f"[✓] Animation saved → {out}")

# ------------------------------------------------------------------ #
# 3 · Rolling MAE plot                                               #
# ------------------------------------------------------------------ #
def rolling_mae_plot(price_df: pd.DataFrame,
                     window: int,
                     out: Path):
    """Plot rolling MAE of persistence baseline."""
    pred = price_df.shift(1)
    mae = (price_df - pred).abs()
    rolling_mae = mae.rolling(window).mean()

    plt.figure(figsize=(9, 5))
    rolling_mae.mean(axis=1).plot()
    plt.title(f'Rolling MAE (window={window} h) – Persistence baseline')
    plt.ylabel('MAE')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[✓] Rolling‑MAE plot saved → {out}")

# ------------------------------------------------------------------ #
# 4 · Dummy synthetic data for demo                                  #
# ------------------------------------------------------------------ #
def _create_dummy_data():
    regions = [f'R{i}' for i in range(10)]
    idx = pd.date_range('2025‑01‑01', periods=72, freq='H')
    rng = np.random.default_rng(42)

    price_df = pd.DataFrame(rng.normal(50, 5, size=(len(idx), len(regions))),
                            index=idx, columns=regions)
    flag_df = pd.DataFrame(
        np.tile((price_df.std(axis=1) < 3).values.reshape(-1, 1),
                (1, len(regions))),
        index=idx, columns=regions)

    G = nx.Graph()
    G.add_nodes_from(regions)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            cap = int(rng.integers(500, 3000))
            G.add_edge(regions[i], regions[j], capacity=cap)

    return G, price_df, flag_df

# ------------------------------------------------------------------ #
# 5 · Main entry                                                     #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Visual analytics toolkit demo')
    p.add_argument('--demo', action='store_true', help='Run demo with dummy data')
    p.add_argument('--outdir', type=str, default='outputs', help='Output directory')
    p.add_argument('--window', type=int, default=24, help='Rolling MAE window (h)')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        G, price_df, flag_df = _create_dummy_data()
    else:
        raise SystemExit('Replace --demo with real data loading logic.')

    tie_line_heatmap(G, price_df, outdir / 'tie_line_heatmap.png')
    animate_price_graph(G, price_df, True,  outdir / 'price_animation.mp4')
    animate_price_graph(G, flag_df, False, outdir / 'coupling_animation.mp4')
    rolling_mae_plot(price_df, args.window, outdir / 'rolling_mae.png')
    print('[✓] Demo finished – visuals available in', outdir)
