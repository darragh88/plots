
"""
decay_regress_diagnostics.py
----------------------------

Utility plots & metrics for analysing the output of an exponentially‑weighted
Bayesian / decay regression.

Functions
~~~~~~~~~
plot_pred_vs_actual(y, y_hat, ax=None)
plot_time_series_overlay(y, y_hat, ax=None)
plot_residuals(y, y_hat, ax=None)
plot_cumulative_error(y, y_hat, ax=None)

plot_beta_paths(beta, feature_names=None, ax=None)
plot_beta_heatmap(beta, feature_names=None, ax=None)
plot_beta_l2_norm(beta, ax=None)
plot_beta_rolling_std(beta, window=48, feature_names=None, ax=None)

plot_beta_prior_diff(beta, beta_prior, feature_names=None, ax=None)
plot_beta_prior_corr(beta, beta_prior, ax=None)

plot_pnl_with_beta_colour(y, y_hat, beta, beta_idx=0, price=None, ax=None)

plot_all(...) – dashboard wrapper
"""

from __future__ import annotations
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  Prediction-quality plots
# ---------------------------------------------------------------------------
def plot_pred_vs_actual(y, y_hat, ax: Optional[plt.Axes] = None):
    y = np.asarray(y); y_hat = np.asarray(y_hat)
    if ax is None: _, ax = plt.subplots()
    ax.scatter(y, y_hat, alpha=0.4, s=8)
    low, high = np.min([y.min(), y_hat.min()]), np.max([y.max(), y_hat.max()])
    ax.plot([low, high], [low, high], lw=2, color="k", alpha=.7)
    ax.set(xlabel="Actual", ylabel="Predicted", title="Predicted vs Actual")
    return ax


def plot_time_series_overlay(y, y_hat, ax: Optional[plt.Axes] = None):
    if ax is None: _, ax = plt.subplots()
    ax.plot(y, label="Actual")
    ax.plot(y_hat, label="Predicted")
    ax.legend(); ax.set_title("Time-series overlay")
    return ax


def plot_residuals(y, y_hat, ax: Optional[plt.Axes] = None):
    resid = np.asarray(y) - np.asarray(y_hat)
    if ax is None: _, ax = plt.subplots()
    ax.plot(resid, lw=.7); ax.axhline(0, lw=.8, ls="--", color="k")
    ax.set_title("Residuals"); return ax


def plot_cumulative_error(y, y_hat, ax: Optional[plt.Axes] = None):
    resid = np.asarray(y) - np.asarray(y_hat)
    if ax is None: _, ax = plt.subplots()
    ax.plot(np.cumsum(resid)); ax.axhline(0, ls="--", lw=.8, color="k")
    ax.set_title("Cumulative error"); return ax


# ---------------------------------------------------------------------------
# 2.  Beta dynamics
# ---------------------------------------------------------------------------
def _default_names(k): return [f"β{j}" for j in range(k)]

def plot_beta_paths(beta, feature_names: Optional[Sequence[str]] = None,
                    ax: Optional[plt.Axes] = None):
    beta = np.asarray(beta); k = beta.shape[0]
    feature_names = feature_names or _default_names(k)
    if ax is None: _, ax = plt.subplots(figsize=(10,3))
    for j in range(k): ax.plot(beta[j], label=feature_names[j])
    ax.axhline(0, lw=.8, color="k"); ax.legend(ncol=min(3,k))
    ax.set_title("Beta paths"); return ax


def plot_beta_heatmap(beta, feature_names: Optional[Sequence[str]] = None,
                      ax: Optional[plt.Axes] = None):
    beta = np.asarray(beta); k = beta.shape[0]
    feature_names = feature_names or _default_names(k)
    lim = np.max(np.abs(beta))
    if ax is None: _, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(beta, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(range(k)); ax.set_yticklabels(feature_names)
    ax.set_title("Beta heat‑map"); plt.colorbar(im, ax=ax); return ax


def plot_beta_l2_norm(beta, ax: Optional[plt.Axes] = None):
    l2 = np.linalg.norm(np.asarray(beta), axis=0)
    if ax is None: _, ax = plt.subplots()
    ax.plot(l2); ax.set_title("||β||₂ over time"); return ax


def plot_beta_rolling_std(beta, window: int = 48,
                          feature_names: Optional[Sequence[str]] = None,
                          ax: Optional[plt.Axes] = None):
    beta = np.asarray(beta); k = beta.shape[0]
    feature_names = feature_names or _default_names(k)
    roll = pd.DataFrame(beta.T).rolling(window).std()
    if ax is None: _, ax = plt.subplots(figsize=(10,3))
    for j in range(k): ax.plot(roll.iloc[:, j], label=feature_names[j])
    ax.set_title(f"Rolling β std (win={window})"); ax.legend(ncol=min(3,k))
    return ax


# ---------------------------------------------------------------------------
# 3.  Prior diagnostics
# ---------------------------------------------------------------------------
def plot_beta_prior_diff(beta, beta_prior,
                         feature_names: Optional[Sequence[str]] = None,
                         ax: Optional[plt.Axes] = None):
    beta = np.asarray(beta); beta_prior = np.asarray(beta_prior)[:, None]
    diff = beta - beta_prior; k = diff.shape[0]
    feature_names = feature_names or _default_names(k)
    if ax is None: _, ax = plt.subplots(figsize=(10,3))
    for j in range(k): ax.plot(diff[j], label=feature_names[j])
    ax.axhline(0, lw=.8, color="k"); ax.legend(ncol=min(3,k))
    ax.set_title("β − β_prior"); return ax


def plot_beta_prior_corr(beta, beta_prior, ax: Optional[plt.Axes] = None):
    beta = np.asarray(beta); beta_prior = np.asarray(beta_prior)
    corr = [np.corrcoef(beta[:, t], beta_prior)[0,1] for t in range(beta.shape[1])]
    if ax is None: _, ax = plt.subplots()
    ax.plot(corr); ax.set_ylim(-1,1); ax.set_title("Corr(β_t, β_prior)")
    return ax


# ---------------------------------------------------------------------------
# 4.  Strategy PnL coloured by a beta sign
# ---------------------------------------------------------------------------
def plot_pnl_with_beta_colour(y, y_hat, beta, beta_idx: int = 0,
                              price: Optional[Sequence[float]] = None,
                              ax: Optional[plt.Axes] = None):
    y = np.asarray(y); y_hat = np.asarray(y_hat)
    if price is not None: y = y / np.asarray(price)
    strat = np.sign(y_hat) * y
    cum = np.cumsum(strat); colour = beta[beta_idx] > 0
    if ax is None: _, ax = plt.subplots()
    ax.scatter(range(len(cum)), cum, c=colour, cmap="coolwarm", s=4)
    ax.axhline(0, lw=.8, ls="--", color="k")
    ax.set_title(f"Cumulative P&L (β{beta_idx} sign colour)"); return ax


# ---------------------------------------------------------------------------
# 5. Dashboard
# ---------------------------------------------------------------------------
def plot_all(y, y_hat, beta, *, feature_names=None, beta_prior=None,
             price=None, rolling_window=48):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14,10), constrained_layout=True)
    gs = gridspec.GridSpec(3,3, figure=fig)

    plot_pred_vs_actual(y,y_hat, ax=fig.add_subplot(gs[0,0]))
    plot_time_series_overlay(y,y_hat, ax=fig.add_subplot(gs[0,1]))
    plot_cumulative_error(y,y_hat, ax=fig.add_subplot(gs[0,2]))

    plot_beta_paths(beta, feature_names, ax=fig.add_subplot(gs[1,:2]))
    plot_beta_heatmap(beta, feature_names, ax=fig.add_subplot(gs[1,2]))

    plot_beta_rolling_std(beta, rolling_window, feature_names,
                          ax=fig.add_subplot(gs[2,0]))
    plot_beta_l2_norm(beta, ax=fig.add_subplot(gs[2,1]))
    plot_pnl_with_beta_colour(y,y_hat,beta,0,price,
                              ax=fig.add_subplot(gs[2,2]))

    fig.suptitle("Decay‑regress diagnostics", fontsize=15)
    return fig
