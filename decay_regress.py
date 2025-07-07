
"""decay_regress.py — Rolling EW Bayesian regression with per‑feature decay.

This module is a near‑drop‑in replacement for the original `decay_regress`
but adds optional per‑feature decay scales via the *decay_scales* keyword.

Author: ChatGPT (refactor for per‑feature forgetting)
"""

import numpy as np
from numbers import Real

# ---------------------------------------------------------------------
# Optional qarray compatibility layer
# ---------------------------------------------------------------------
try:
    import qarray                             # your real helper lib
except ImportError:
    class qarray:
        @staticmethod
        def set_array(a, idx, val):
            a = a.copy()
            a[idx] = val
            return a

        @staticmethod
        def gt(x, y):
            return np.greater(x, y)

        @staticmethod
        def geq(x, y):
            return np.greater_equal(x, y)

        @staticmethod
        def is_positive_definite(m):
            return np.all(np.linalg.eigvals(m) > 0)

DT_DOUBLE = np.float64


# ---------------------------------------------------------------------
def decay_regress(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prior_mean:   np.ndarray,
    prior_covar:  np.ndarray,
    decay_scale:  float        = np.Inf,
    decay_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Parallel rolling regression with exponential decay and
    Bayesian prior.

    Parameters
    ----------
    x : K × T × N ndarray
    y : T × N ndarray
    prior_mean : (K,) ndarray
    prior_covar : (K, K) ndarray
    decay_scale : float, optional
        Common decay if *decay_scales* is not provided.
    decay_scales : (K,) array_like, optional
        Per‑feature time constants τ_k.  Overrides *decay_scale* for
        x‑related moments.

    Returns
    -------
    beta : K × T × N ndarray of posterior coefficients.
    """
    # ---- validation --------------------------------------------------
    for arr in (x, y, prior_mean, prior_covar):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype.kind in ('f', 'i')
        assert not np.any(np.isinf(arr))

    assert x.ndim == 3
    K, T, N = x.shape
    assert y.shape == (T, N)
    assert prior_mean.shape == (K,)
    assert prior_covar.shape == (K, K)
    assert qarray.is_positive_definite(prior_covar)

    if decay_scales is None:
        assert isinstance(decay_scale, Real) and qarray.gt(decay_scale, 0)
        decay_vec = np.exp(-1.0 / decay_scale) * np.ones(K, DT_DOUBLE)
    else:
        decay_scales = np.asarray(decay_scales, dtype=DT_DOUBLE)
        assert decay_scales.shape == (K,)
        assert np.all(qarray.gt(decay_scales, 0))
        decay_vec = np.exp(-1.0 / decay_scales)

    decay_y = (np.exp(-1.0 / decay_scale)
               if np.isfinite(decay_scale) else 1.0)

    decay_outer = decay_vec[:, None] * decay_vec[None, :]

    # ---- initialise --------------------------------------------------
    prior_precision = np.linalg.inv(prior_covar)

    bad_x = np.isnan(x)
    bad   = bad_x.any(axis=0) | np.isnan(y)
    w     = (~bad).astype(DT_DOUBLE)

    x0 = qarray.set_array(x, bad_x, 0)
    y0 = qarray.set_array(y, bad,   0)

    sw, swy2     = (np.zeros(N,         DT_DOUBLE) for _ in range(2))
    swx2         = np.zeros((K, K, N),  DT_DOUBLE)
    swxy         = np.zeros((K,     N), DT_DOUBLE)

    beta         = np.empty_like(x, dtype=DT_DOUBLE)
    beta_init    = np.tile(prior_mean[:, None], (1, N)).T  # N × K

    # -----------------------------------------------------------------
    def _iterate_beta(beta_a: np.ndarray) -> np.ndarray:
        """One posterior update, vectorised over N series."""
        xy  = swxy.T[:, None, :]      # N × 1 × K
        xx  = swx2.T                 # N × K × K
        bt  = beta_a[:, :, None]
        btt = beta_a[:, None, :]

        with np.errstate(invalid='ignore'):
            res_var0 = (swy2
                        + (-2 * (xy @ bt) + btt @ xx @ bt)[:, 0, 0]) / sw

        good = np.nonzero(qarray.geq(res_var0, 0))[0]
        res_var = np.where(qarray.geq(res_var0, 0), res_var0, 1e-5)

        lam = res_var[:, None, None] * prior_precision
        A   = xx + lam
        b   = swxy.T + lam @ prior_mean

        beta_new = beta_a.copy()
        beta_new[good] = np.linalg.solve(A[good], b[good])
        return beta_new
    # -----------------------------------------------------------------

    for t in range(T):
        # decay moments
        sw    *= decay_y
        swy2  *= decay_y
        swxy  *= decay_vec[:, None]
        swx2  *= decay_outer[:, :, None]

        # accumulate new point
        sw    += w[t]
        swy2  += w[t] * np.square(y0[t])
        swxy  += w[t] * y0[t] * x0[:, t]
        swx2  += w[t] * x0[:, t] * x0[:, t][:, None]

        # two-pass update
        beta0      = _iterate_beta(beta_init)
        beta1      = _iterate_beta(beta0)
        beta[:, t] = beta1.T

    return beta
