
"""
jepx_plots_elasticity.py
========================
Matplotlib helpers for visualising supply- and demand-side elasticity
based on the BidCurve / MultiBidCurve containers defined in jepx_loader.py.

Public functions
----------------
plot_elasticity_surface(container, *, side='supply')
plot_elasticity_curve(container, ts, *, side='both')
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal, Union

# import your existing containers & stats helpers
from jepx_loader import BidCurve, MultiBidCurve
from jepx_stats import elasticity_panel, elasticity


# ------------------------------------------------------------------
# Heat‑map of dV/dP across all timeslots
# ------------------------------------------------------------------
def plot_elasticity_surface(
    container: Union[BidCurve, MultiBidCurve],
    *,
    side: Literal['supply', 'demand'] = 'supply',
):
    """Heat‑map: rows = time, columns = price, colour = dV/dP."""
    surf = elasticity_panel(container, side=side)  # DataFrame

    fig, ax = plt.subplots()
    im = ax.imshow(
        surf.values,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
    )

    ax.set_xlabel('Price bins [¥/kWh]')
    ax.set_ylabel('Time' if isinstance(container, MultiBidCurve) else 'Time code (1‑48)')
    ax.set_title(f'{side.title()}‑side elasticity surface')

    # Sparse ticks for readability
    ax.set_xticks(np.linspace(0, len(surf.columns) - 1, 6))
    ax.set_xticklabels(
        np.round(np.linspace(surf.columns[0], surf.columns[-1], 6), 2)
    )

    if isinstance(container, MultiBidCurve):
        yticks = np.linspace(0, len(surf.index) - 1, 5, dtype=int)
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            pd.to_datetime(surf.index[yticks]).strftime('%m-%d\n%H:%M')
        )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('dV / dP [MWh per ¥/kWh]')
    ax.grid(False)
    return fig


# ------------------------------------------------------------------
# Elasticity curve for a single timeslot
# ------------------------------------------------------------------
def plot_elasticity_curve(
    container: Union[BidCurve, MultiBidCurve],
    ts: Union[str, pd.Timestamp, int],
    *,
    side: Literal['supply', 'demand', 'both'] = 'both',
):
    """Line plot of dV/dP vs price for one 30‑minute auction."""
    if isinstance(container, MultiBidCurve):
        curve_slice = container[ts]
        label = str(pd.to_datetime(ts))
    else:
        curve_slice = container.slice_time(int(ts))
        label = f'time‑code {ts}'

    fig, ax = plt.subplots()

    if side in {'supply', 'both'}:
        sup = elasticity(curve_slice, side='supply')
        ax.plot(sup.index, sup.values, label='Supply', linewidth=1.2)

    if side in {'demand', 'both'}:
        dem = elasticity(curve_slice, side='demand')
        ax.plot(dem.index, dem.values, label='Demand', linewidth=1.2)

    ax.set_xlabel('Price [¥/kWh]')
    ax.set_ylabel('dV / dP [MWh per ¥/kWh]')
    ax.set_title(f'Elasticity curve — {label}')
    ax.grid(True, alpha=0.3)
    if side == 'both':
        ax.legend()
    return fig
