
"""
jepx_uniform.py
===============
Uniform‑price cost helpers for JEPX day‑ahead auctions.

(Requires jepx_stats.py and jepx_loader.py in PYTHONPATH)
"""

from __future__ import annotations
from typing import Tuple, Literal, Union
import numpy as np
import pandas as pd

from jepx_stats import clearing_price, clearing_demand, _iter_slices
from jepx_loader import BidCurve, MultiBidCurve


# ————————————————————————————————————————————————
def trading_cost_uniform(slice_df: pd.DataFrame,
                         vol_mwh: float,
                         *,
                         side: Literal['buy', 'sell'] = 'buy') -> Tuple[float, float]:
    """Total cost ¥ and clearing price ¥/kWh for a uniform‑price auction."""
    if vol_mwh <= 0:
        raise ValueError('vol_mwh must be positive')

    price = slice_df.index.to_numpy(dtype=float)
    supply = slice_df['supply_cum'].to_numpy(dtype=float)
    demand = slice_df['demand_cum'].to_numpy(dtype=float)

    if side == 'buy':
        if vol_mwh > supply[-1]:
            raise ValueError('Not enough supply depth')
        cp = np.interp(vol_mwh, supply, price)
    else:
        if vol_mwh > demand[-1]:
            raise ValueError('Not enough demand depth')
        cp = np.interp(vol_mwh, demand, price)

    total = vol_mwh * cp * 1_000
    return float(total), float(cp)


def marginal_price_uniform(slice_df: pd.DataFrame,
                           *,
                           extra_vol_mwh: float = 0.5,
                           side: Literal['buy', 'sell'] = 'buy') -> Tuple[float, float]:
    """Incremental cost ¥ and marginal price ¥/kWh beyond clearing."""
    if extra_vol_mwh <= 0:
        raise ValueError('extra_vol_mwh must be positive')

    cv0 = clearing_demand(slice_df)
    cp0 = clearing_price(slice_df)

    _, cp1 = trading_cost_uniform(slice_df, cv0 + extra_vol_mwh, side=side)

    incr_cost = (cv0 + extra_vol_mwh) * cp1 * 1_000 - cv0 * cp0 * 1_000
    incr_price = incr_cost / (extra_vol_mwh * 1_000)
    return float(incr_cost), float(incr_price)


def trading_cost_uniform_series(container: Union[BidCurve, MultiBidCurve],
                                vol_mwh: float,
                                *,
                                side: Literal['buy', 'sell'] = 'buy') -> pd.DataFrame:
    """Vectorised total cost and uniform clearing price per timeslot."""
    labels, costs, cps = [], [], []
    for lbl, sl in _iter_slices(container):
        c, p = trading_cost_uniform(sl, vol_mwh, side=side)
        labels.append(lbl); costs.append(c); cps.append(p)

    idx_name = 'timestamp' if isinstance(container, MultiBidCurve) else 'time_code'
    return pd.DataFrame({'total_cost': costs, 'clearing_price': cps},
                        index=pd.Index(labels, name=idx_name))


def marginal_price_uniform_series(container: Union[BidCurve, MultiBidCurve],
                                  *,
                                  extra_vol_mwh: float = 0.5,
                                  side: Literal['buy', 'sell'] = 'buy') -> pd.DataFrame:
    """Vectorised incremental cost and marginal price per timeslot."""
    labels, inc_c, inc_p = [], [], []
    for lbl, sl in _iter_slices(container):
        c, p = marginal_price_uniform(sl, extra_vol_mwh=extra_vol_mwh, side=side)
        labels.append(lbl); inc_c.append(c); inc_p.append(p)

    idx_name = 'timestamp' if isinstance(container, MultiBidCurve) else 'time_code'
    return pd.DataFrame({'incremental_cost': inc_c, 'marginal_price': inc_p},
                        index=pd.Index(labels, name=idx_name))
