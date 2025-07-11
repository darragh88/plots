
"""
jepx_marginal.py
================
Helpers for computing the incremental (marginal) cost or revenue of
buying/selling an additional volume beyond the market-clearing point.

Depends on:
    * jepx_stats.py  (for _clearing_price_volume, trading_cost, _iter_slices)
    * jepx_loader.py (for BidCurve, MultiBidCurve)
"""

from __future__ import annotations

from typing import Tuple, Literal, Union

import pandas as pd

# import private helpers from stats module
from jepx_stats import (
    _clearing_price_volume,
    trading_cost,
    _iter_slices,
)
from jepx_loader import BidCurve, MultiBidCurve


# ------------------------------------------------------------------
# single‑slice incremental cost / price
# ------------------------------------------------------------------
def marginal_price(
    slice_df: pd.DataFrame,
    *,
    extra_vol_mwh: float = 1.0,
    side: Literal['buy', 'sell'] = 'buy',
) -> Tuple[float, float]:
    """Return (incremental_cost ¥, marginal_price ¥/kWh) for one slice."""

    if extra_vol_mwh <= 0:
        raise ValueError('extra_vol_mwh must be positive')

    # clearing stats
    cp, cv = _clearing_price_volume(slice_df)

    # cost / revenue up to clearing
    cleared_cost, _ = trading_cost(slice_df, cv, side=side)

    # cost / revenue for clearing + ΔQ
    new_cost, _ = trading_cost(slice_df, cv + extra_vol_mwh, side=side)

    incr_cost  = new_cost - cleared_cost                     # ¥
    incr_price = incr_cost / (extra_vol_mwh * 1_000)         # ¥/kWh

    return incr_cost, incr_price


# ------------------------------------------------------------------
# vectorised across all timeslots
# ------------------------------------------------------------------
def marginal_price_series(
    container: Union[BidCurve, MultiBidCurve],
    *,
    extra_vol_mwh: float = 1.0,
    side: Literal['buy', 'sell'] = 'buy',
    return_dataframe: bool = True,
):
    """
    Run marginal_price() across every 30‑min auction in the container.

    Returns
    -------
    DataFrame (default) with columns:
        incremental_cost   — ¥
        marginal_price     — ¥/kWh

    Index
    -----
        1‑48 for BidCurve  | pd.Timestamp for MultiBidCurve
    """
    labels, inc_costs, inc_prices = [], [], []

    for lbl, sl in _iter_slices(container):
        cost, price = marginal_price(
            sl, extra_vol_mwh=extra_vol_mwh, side=side
        )
        labels.append(lbl)
        inc_costs.append(cost)
        inc_prices.append(price)

    idx_name = 'timestamp' if isinstance(container, MultiBidCurve) else 'time_code'
    cost_s  = pd.Series(inc_costs, index=labels, name='incremental_cost')
    price_s = pd.Series(inc_prices, index=labels, name='marginal_price')

    if return_dataframe:
        return pd.concat([cost_s, price_s], axis=1).rename_axis(idx_name)

    return cost_s, price_s
