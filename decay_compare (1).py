"""
decay_compare.py – complete decay-regression toolkit + covariance-sweep helper
==============================================================================

MIT License 2025
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ------------------------------------------------------------------ #
# 0) Small helpers                                                   #
# ------------------------------------------------------------------ #
def _split_numeric_cat(
    df: pd.DataFrame,
    features: list[str],
    treat_binary_as_cat: bool = True,
):
    """Return numeric_cols, categorical_cols."""
    num, cat = [], []
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            uniq = set(df[c].dropna().unique())
            if treat_binary_as_cat and uniq.issubset({0, 1}):
                cat.append(c)
            else:
                num.append(c)
        else:
            cat.append(c)
    return num, cat


def _encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Label-encode categoricals; NaNs become -1."""
    if not cat_cols:
        return pd.DataFrame(index=df.index)
    return df[cat_cols].apply(lambda s: s.astype("category").cat.codes)


# ------------------------------------------------------------------ #
# 1) Standardisation / unstandardisation                             #
# ------------------------------------------------------------------ #
def identity_standardize(df: pd.DataFrame, features: list[str], target: str):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing: {missing}")

    T, N = len(df), 1
    feat_TN = df[features].to_numpy(dtype=float).T[:, :, None]
    x = np.concatenate((np.ones((1, T, N)), feat_TN), axis=0)
    y = df[target].to_numpy(dtype=float).reshape(T, N)
    return x, y, np.zeros(0), np.ones(0)


def z_standardize(df: pd.DataFrame, features: list[str], target: str):
    numeric, categorical = _split_numeric_cat(df, features)
    order = numeric + categorical
    q = len(numeric)

    if q:
        means = df[numeric].mean().to_numpy()
        stds = df[numeric].std().to_numpy() + 1e-8
        scaled_num = (df[numeric] - means) / stds
    else:
        means, stds = np.zeros(0), np.ones(0)
        scaled_num = pd.DataFrame(index=df.index)

    cat_enc = _encode_categoricals(df, categorical)
    combined = pd.concat([scaled_num, cat_enc], axis=1)[order]

    feat_TN = combined.to_numpy(dtype=float).T[:, :, None]
    T, N = len(df), 1
    x = np.concatenate((np.ones((1, T, N)), feat_TN), axis=0)
    y = df[target].to_numpy(dtype=float).reshape(T, N)
    return x, y, means, stds


def z_unstandardize(betas: np.ndarray, means: np.ndarray, stds: np.ndarray):
    q, T = len(means), betas.shape[1]
    N = 1
    slopes = betas[1 : q + 1] / stds[:, None, None] if q else np.zeros((0, T, N))
    intercept = (
        betas[0] - (betas[1 : q + 1] * (means / stds)[:, None, None]).sum(0)
        if q
        else betas[0]
    )
    return np.concatenate([intercept[None], slopes, betas[q + 1 :]], axis=0)


def ewm_standardize(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    halflife: float,
    burnin_steps: int,
):
    numeric, categorical = _split_numeric_cat(df, features)
    order = numeric + categorical
    q, T = len(numeric), len(df)

    if q:
        means_df = df[numeric].ewm(halflife=halflife, adjust=False).mean()
        stds_df = df[numeric].ewm(halflife=halflife, adjust=False).std() + 1e-8
        means_df.iloc[:burnin_steps] = np.nan
        stds_df.iloc[:burnin_steps] = np.nan
        means_df, stds_df = means_df.bfill(), stds_df.bfill()
        means, stds = means_df.to_numpy().T, stds_df.to_numpy().T
        scaled_num = (df[numeric] - means_df) / stds_df
    else:
        means, stds = np.zeros((0, T)), np.ones((0, T))
        scaled_num = pd.DataFrame(index=df.index)

    cat_enc = _encode_categoricals(df, categorical)
    combined = pd.concat([scaled_num, cat_enc], axis=1)[order]
    feat_TN = combined.to_numpy(dtype=float).T[:, :, None]
    x = np.concatenate((np.ones((1, T, 1)), feat_TN), axis=0)
    y = df[target].to_numpy(dtype=float).reshape(T, 1)
    return x, y, means, stds


def ewm_unstandardize(betas: np.ndarray, means: np.ndarray, stds: np.ndarray):
    q, T = means.shape
    N = 1
    slopes = betas[1 : q + 1] / stds[:, :, None] if q else np.zeros((0, T, N))
    intercept = (
        betas[0] - (betas[1 : q + 1] * (means / stds)[:, :, None]).sum(0)
        if q
        else betas[0]
    )
    return np.concatenate([intercept[None], slopes, betas[q + 1 :]], axis=0)


# ------------------------------------------------------------------ #
# 2) Core decay regression                                           #
# ------------------------------------------------------------------ #
def decay_regress(
    x: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_covar: np.ndarray,
    decay_scale: float,
):
    """Sequential Kalman-style filtering; observation variance fixed to 1."""
    K, T = x.shape[0], x.shape[1]
    beta_mean = prior_mean.reshape(K, 1)
    beta_covar = prior_covar.copy()
    Q = np.eye(K) * decay_scale
    betas = np.zeros((K, T, 1))

    for t in range(T):
        xt = x[:, t : t + 1, 0]
        yt = y[t : t + 1]
        beta_covar_pred = beta_covar + Q
        S = (xt.T @ beta_covar_pred @ xt) + 1.0
        Kt = (beta_covar_pred @ xt) / S
        resid = yt - xt.T @ beta_mean
        beta_mean = beta_mean + Kt * resid
        beta_covar = (np.eye(K) - Kt @ xt.T) @ beta_covar_pred
        betas[:, t, 0] = beta_mean[:, 0]
    return betas


# ------------------------------------------------------------------ #
# 3) Diagnostics (minimal)                                           #
# ------------------------------------------------------------------ #
def _plot_model_diagnostics(
    y: np.ndarray,
    yhat: np.ndarray,
    resid: np.ndarray,
    betas_unscaled: np.ndarray,
    features: list[str],
):
    plt.figure(figsize=(6, 3))
    plt.plot(resid, ".", ms=3)
    plt.axhline(0, c="k", lw=0.7)
    plt.title("Residuals")
    plt.show()

    sm.qqplot(resid, line="s")
    plt.title("Q-Q")
    plt.show()

    names = ["Intercept"] + features
    for i, name in enumerate(names):
        plt.figure(figsize=(6, 3))
        plt.plot(betas_unscaled[i, :, 0])
        plt.title(name)
        plt.show()


# ------------------------------------------------------------------ #
# 4) High-level pipeline                                             #
# ------------------------------------------------------------------ #
def decay_pipeline(
    df: pd.DataFrame,
    features: list[str] | str,
    target: str,
    scaling: str = "identity",
    scale_kwargs: dict | None = None,
    backshift: int = 48,
    tau: float = 10.0,
    decay_scale: float = 2880.0,
    plot: bool = False,
):
    if isinstance(features, str):
        features = [features]
    missing = [c for c in features if c not in df.columns]
    if missing or target not in df.columns:
        raise ValueError(f"Missing columns: {missing}")

    if scaling == "identity":
        x, y, means, stds = identity_standardize(df, features, target)
    elif scaling == "ewm":
        if not scale_kwargs or not {"halflife", "burnin_steps"} <= scale_kwargs.keys():
            raise ValueError("scale_kwargs must include 'halflife' and 'burnin_steps'")
        x, y, means, stds = ewm_standardize(
            df, features, target, **scale_kwargs
        )
    else:
        raise ValueError("scaling must be 'identity' or 'ewm'")

    K = x.shape[0]
    betas = decay_regress(
        x,
        y,
        prior_mean=np.zeros(K),
        prior_covar=np.eye(K) * tau,
        decay_scale=decay_scale,
    )
    betas_lag = np.roll(betas, backshift, axis=1)
    betas_lag[:, :backshift] = np.nan
    yhat = np.nansum(x * betas_lag, axis=0)
    resid = yhat.flatten() - y.flatten()
    mse = np.nanmean(resid**2)
    r2 = 1 - np.nansum(resid**2) / np.nansum(
        (y.flatten() - np.nanmean(y)) ** 2
    )

    if scaling == "identity":
        betas_orig = betas_lag
    elif scaling == "ewm":
        betas_orig = ewm_unstandardize(betas_lag, means, stds)

    if plot:
        _plot_model_diagnostics(y.flatten(), yhat.flatten(), resid, betas_orig, features)

    return dict(
        betas=betas_lag,
        betas_orig=betas_orig,
        yhat=yhat,
        resid=resid,
        mse=mse,
        r2=r2,
    )


# ------------------------------------------------------------------ #
# 5) Covariance-sweep comparison                                     #
# ------------------------------------------------------------------ #
def _fit_once(
    df: pd.DataFrame,
    feat_list: list[str],
    target: str,
    scaling: str,
    scale_kwargs: dict | None,
    tau: float,
    decay_scale: float,
    backshift: int,
):
    return decay_pipeline(
        df,
        feat_list,
        target,
        scaling=scaling,
        scale_kwargs=scale_kwargs,
        backshift=backshift,
        tau=tau,
        decay_scale=decay_scale,
        plot=False,
    )


def compare_covariance(
    df: pd.DataFrame,
    features: list[str] | str,
    target: str,
    add_feature: str | None = None,
    tau_base: float = 10.0,
    tau_list_alt: list[float] | None = None,
    decay_scale: float = 2880.0,
    scaling: str = "identity",
    scale_kwargs: dict | None = None,
    backshift: int = 48,
):
    if isinstance(features, str):
        features = [features]

    baseline = _fit_once(
        df,
        features,
        target,
        scaling,
        scale_kwargs,
        tau_base,
        decay_scale,
        backshift,
    )

    if add_feature is None:
        return dict(baseline_tau=tau_base, baseline=baseline, alternatives={})

    if tau_list_alt is None:
        tau_list_alt = [tau_base / 10, tau_base, tau_base * 10]

    full_feats = features + [add_feature]
    alt = {
        tau: _fit_once(
            df,
            full_feats,
            target,
            scaling,
            scale_kwargs,
            tau,
            decay_scale,
            backshift,
        )
        for tau in tau_list_alt
    }

    return dict(baseline_tau=tau_base, baseline=baseline, alternatives=alt)


# ------------------------------------------------------------------ #
# 6) Plot R² vs τ                                                    #
# ------------------------------------------------------------------ #
def plot_r2_vs_tau(result: dict):
    base_tau = result["baseline_tau"]
    alt = result.get("alternatives", {})
    taus = [base_tau] + sorted(alt.keys())
    r2s = [result["baseline"]["r2"]] + [alt[t]["r2"] for t in sorted(alt.keys())]

    plt.figure(figsize=(6, 4))
    plt.plot(taus, r2s, marker="o")
    plt.xscale("log")
    plt.xlabel("Prior variance scale τ (log)")
    plt.ylabel("R²")
    plt.title("Predictive R² vs prior covariance scale")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()


if __name__ == "__main__":
    print("Module loaded. Import decay_compare and use its functions.")
