"""
sequential_forecaster.py
────────────────────────
A single-file, model-agnostic framework for sequential / online forecasting
on a **pandas DataFrame with a DatetimeIndex**.  The file contains:

1.  WindowProvider   – hands out the history visible at “now”.
2.  ModelAdapters    – OLS, SGD, (S)VR, tree ensembles, LightGBM, XGBoost,
                       CatBoost.  All share one interface.
3.  Update policies  – choose when to do nothing / partial update / full
                       retrain.
4.  Forecaster       – orchestrates the time loop and returns R², MSE,
                       ŷ-series, and y-series.

Combine any adapter with any policy; plug straight into your already-scaled
DataFrame.  Requires *only* packages that you listed as available.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# 0)  stdlib / common imports
# ─────────────────────────────────────────────────────────────
import abc
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR

# optional external libraries (only imported if installed)
try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    CatBoostRegressor = None

# ─────────────────────────────────────────────────────────────
# 1)  WindowProvider  – history visible at “now”
# ─────────────────────────────────────────────────────────────
class WindowProvider:
    """
    Parameters
    ----------
    df        : pandas DataFrame (already scaled / encoded, DatetimeIndex)
    lookback  : - None  → expanding window from the first row
                - str   → pandas offset alias (e.g. "30D", "72h")
                - int   → fixed number of rows
    horizon   : how many **rows** ahead you want the prediction.
                horizon = 0  → same-timestamp prediction (“now-cast”)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: Optional[str | int] = None,
        horizon: int = 0,
    ):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        self.df = df.sort_index()
        self.lookback = lookback
        self.h = horizon

    # ---------------------------------------------------------
    def _history_slice(self, now: pd.Timestamp) -> pd.DataFrame:
        """Return df[..now] subject to look-back rule."""
        if self.lookback is None:
            return self.df.loc[:now]

        if isinstance(self.lookback, int):
            loc_now = self.df.index.get_loc(now)
            start_pos = max(0, loc_now - self.lookback + 1)
            return self.df.iloc[start_pos : loc_now + 1]

        start_time = now - pd.Timedelta(self.lookback)
        return self.df.loc[start_time:now]

    # ---------------------------------------------------------
    def get_train_data(
        self,
        now: pd.Timestamp,
        features: list[str],
        target: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns X_history  (features at t-k … t)
                y_history  (future target t + horizon)
        so that each row of X aligns with its target row.
        """
        hist = self._history_slice(now)

        # shift target *up* by horizon rows unless horizon == 0
        if self.h == 0:
            y_shifted = hist[target]
        else:
            y_shifted = hist[target].shift(-self.h)

        valid = y_shifted.notna()
        X_train = hist[features][valid]
        y_train = y_shifted[valid]
        return X_train, y_train

    # ---------------------------------------------------------
    def get_pred_row(
        self, now: pd.Timestamp, features: list[str]
    ) -> pd.DataFrame:
        """Row fed into adapter.predict() at issue-time *now*."""
        return self.df.loc[[now], features]

    # ---------------------------------------------------------
    def truth_at_horizon(
        self, now: pd.Timestamp, target: str
    ) -> Optional[float]:
        """Ground-truth y[now + horizon]."""
        idx, pos = self.df.index, self.df.index.get_loc(now)
        tgt_pos = pos + self.h
        if tgt_pos >= len(idx):
            return None
        return self.df[target].iloc[tgt_pos]


# ─────────────────────────────────────────────────────────────
# 2)  Base & concrete ModelAdapters
# ─────────────────────────────────────────────────────────────
class BaseAdapter(abc.ABC):
    """Shared interface: predict / full_retrain / optional partial_update."""

    def __init__(self):
        self._is_fitted = False
        self.last_full_retrain: Optional[pd.Timestamp] = None

    # ---- prediction ----------------------------------------
    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not yet fitted")
        return self._predict_impl(X)

    @abc.abstractmethod
    def _predict_impl(self, X): ...

    # ---- full retrain -------------------------------------
    def full_retrain(self, X, y):
        self._full_retrain_impl(X, y)
        self._is_fitted = True

    @abc.abstractmethod
    def _full_retrain_impl(self, X, y): ...

    # ---- incremental update (optional) --------------------
    def partial_update(self, X, y):
        """Default: do nothing (override in incremental adapters)."""
        return


# -- Linear OLS ------------------------------------------------------------
class OLSAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def _predict_impl(self, X):
        return self.model.predict(X)

    def _full_retrain_impl(self, X, y):
        self.model = LinearRegression()          # fresh instance
        self.model.fit(X, y)


# -- Incremental SGD -------------------------------------------------------
class SGDAdapter(BaseAdapter):
    def __init__(self, **sgd_kwargs):
        super().__init__()
        self._sgd_kwargs = {"loss": "squared_loss", **sgd_kwargs}
        self.model = SGDRegressor(**self._sgd_kwargs)

    def _predict_impl(self, X):
        return self.model.predict(X)

    def _full_retrain_impl(self, X, y):
        self.model = SGDRegressor(**self._sgd_kwargs)
        self.model.fit(X, y)

    def partial_update(self, X, y):
        self.model.partial_fit(X, y)
        self._is_fitted = True


# -- Helper mix-in for models with no partial_fit --------------------------
class _BatchOnlyAdapter(BaseAdapter):
    def partial_update(self, X, y):
        return  # silently ignore if policy asks for partial


# -- Decision Tree ---------------------------------------------------------
class DecisionTreeAdapter(_BatchOnlyAdapter):
    def __init__(self, **tree_kwargs):
        super().__init__()
        self._tree_kwargs = tree_kwargs
        self.model = DecisionTreeRegressor(**tree_kwargs)

    def _predict_impl(self, X):
        return self.model.predict(X)

    def _full_retrain_impl(self, X, y):
        self.model = DecisionTreeRegressor(**self._tree_kwargs)
        self.model.fit(X, y)


# -- Random Forest ---------------------------------------------------------
class RandomForestAdapter(_BatchOnlyAdapter):
    def __init__(self, **rf_kwargs):
        super().__init__()
        self._rf_kwargs = rf_kwargs
        self.model = RandomForestRegressor(**rf_kwargs)

    def _predict_impl(self, X):
        return self.model.predict(X)

    def _full_retrain_impl(self, X, y):
        self.model = RandomForestRegressor(**self._rf_kwargs)
        self.model.fit(X, y)


# -- Gradient Boosting -----------------------------------------------------
class GradientBoostingAdapter(_BatchOnlyAdapter):
    def __init__(self, **gb_kwargs):
        super().__init__()
        self._gb_kwargs = gb_kwargs
        self.model = GradientBoostingRegressor(**gb_kwargs)

    def _predict_impl(self, X):
        return self.model.predict(X)

    def _full_retrain_impl(self, X, y):
        self.model = GradientBoostingRegressor(**self._gb_kwargs)
        self.model.fit(X, y)


# -- SVR (kernel SVM regression) ------------------------------------------
class SVRAdapter(_BatchOnlyAdapter):
    def __init__(self, **svr_kwargs):
        super().__init__()
        self._svr_kwargs = svr_kwargs
        self.model = SVR(**svr_kwargs)

    def _predict_impl(self, X):
        return self.model.predict(X)

    def _full_retrain_impl(self, X, y):
        self.model = SVR(**self._svr_kwargs)
        self.model.fit(X, y)


# -- LightGBM --------------------------------------------------------------
if lgb is not None:

    class LightGBMAdapter(_BatchOnlyAdapter):
        def __init__(self, **lgbm_kwargs):
            super().__init__()
            self._lgbm_kwargs = lgbm_kwargs
            self.model = lgb.LGBMRegressor(**lgbm_kwargs)

        def _predict_impl(self, X):
            return self.model.predict(X)

        def _full_retrain_impl(self, X, y):
            self.model = lgb.LGBMRegressor(**self._lgbm_kwargs)
            self.model.fit(X, y)


# -- XGBoost ---------------------------------------------------------------
if XGBRegressor is not None:

    class XGBoostAdapter(_BatchOnlyAdapter):
        def __init__(self, **xgb_kwargs):
            super().__init__()
            self._xgb_kwargs = xgb_kwargs
            self.model = XGBRegressor(**xgb_kwargs)

        def _predict_impl(self, X):
            return self.model.predict(X)

        def _full_retrain_impl(self, X, y):
            self.model = XGBRegressor(**self._xgb_kwargs)
            self.model.fit(X, y)


# -- CatBoost --------------------------------------------------------------
if CatBoostRegressor is not None:

    class CatBoostAdapter(_BatchOnlyAdapter):
        def __init__(self, **cb_kwargs):
            super().__init__()
            self._cb_kwargs = {"verbose": 0, **cb_kwargs}
            self.model = CatBoostRegressor(**self._cb_kwargs)

        def _predict_impl(self, X):
            return self.model.predict(X)

        def _full_retrain_impl(self, X, y):
            self.model = CatBoostRegressor(**self._cb_kwargs)
            self.model.fit(X, y)


# ─────────────────────────────────────────────────────────────
# 3)  Update-decision policies
# ─────────────────────────────────────────────────────────────
class UpdatePolicy(abc.ABC):
    @abc.abstractmethod
    def decide(
        self, adapter: BaseAdapter, now: pd.Timestamp
    ) -> Literal["none", "partial", "full"]:
        ...


class AlwaysPartial(UpdatePolicy):
    """Partial update every step if adapter supports it."""

    def decide(self, adapter, now):
        supports_partial = (
            adapter.partial_update.__qualname__.split(".")[0]
            != "BaseAdapter"
        )
        return "partial" if supports_partial else "none"


class PeriodicRetrain(UpdatePolicy):
    """Full retrain every *period*; nothing in between."""

    def __init__(self, period: str | pd.Timedelta):
        self.period = pd.Timedelta(period)

    def decide(self, adapter, now):
        if adapter.last_full_retrain is None:
            return "full"
        if now - adapter.last_full_retrain >= self.period:
            return "full"
        return "none"


class Hybrid(UpdatePolicy):
    """Partial every step; full when *period* elapsed."""

    def __init__(self, period: str | pd.Timedelta):
        self.period = pd.Timedelta(period)

    def decide(self, adapter, now):
        if adapter.last_full_retrain is None:
            return "full"
        if now - adapter.last_full_retrain >= self.period:
            return "full"
        supports_partial = (
            adapter.partial_update.__qualname__.split(".")[0]
            != "BaseAdapter"
        )
        return "partial" if supports_partial else "none"


# ─────────────────────────────────────────────────────────────
# 4)  Forecaster – orchestrates the time loop
# ─────────────────────────────────────────────────────────────
@dataclass
class Forecaster:
    df: pd.DataFrame
    features: list[str]
    target: str
    adapter: BaseAdapter
    window: WindowProvider
    policy: UpdatePolicy
    horizon: int = 0  # rows ahead; must match WindowProvider.h

    # ---------------------------------------------------------
    def run(self):
        if self.window.h != self.horizon:
            raise ValueError("Forecaster.horizon and WindowProvider.h must match")

        idx = self.df.index
        preds, truths, times = [], [], []

        # start once there is at least one y_{t+horizon}
        start_pos = self.horizon
        end_pos = len(idx) - self.horizon  # last usable issue time

        for pos in range(start_pos, end_pos):
            now = idx[pos]

            # 1. decide update action
            action = self.policy.decide(self.adapter, now)

            # 2. history visible at now
            X_hist, y_hist = self.window.get_train_data(
                now, self.features, self.target
            )

            if action == "full":
                self.adapter.full_retrain(X_hist, y_hist)
                self.adapter.last_full_retrain = now
            elif action == "partial":
                # newest aligned row (features at pos-h, target at pos)
                X_inc = X_hist.tail(1)
                y_inc = y_hist.tail(1)
                self.adapter.partial_update(X_inc, y_inc)

            # 3. predict ŷ[t + h] using features at time now
            X_now = self.window.get_pred_row(now, self.features)
            y_hat = float(self.adapter.predict(X_now)[0])
            preds.append(y_hat)
            times.append(now)

            # 4. grab ground-truth y[t + h]
            truth = self.window.truth_at_horizon(now, self.target)
            truths.append(truth)

        # build aligned series
        y_pred_series = pd.Series(preds, index=times, name="y_pred")
        y_true_series = pd.Series(truths, index=times, name="y_true")

        mask = y_true_series.notna()
        y_pred_c, y_true_c = y_pred_series[mask], y_true_series[mask]

        mse = mean_squared_error(y_true_c, y_pred_c)
        r2 = r2_score(y_true_c, y_pred_c)

        return {
            "r2": r2,
            "mse": mse,
            "y_pred": y_pred_series,
            "y_true": y_true_series,
        }
