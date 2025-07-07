
"""decay_regress_geom.py — Rolling EW Bayesian regression with per-feature
decay using the *geometric‑mean* rule for cross‑moments.

If all decay_scales are identical the behaviour is exactly the same
as the original single‑decay implementation.
"""

import numpy as np
from numbers import Real

# ---------------------------------------------------------------------
# Lightweight stubs for qarray helpers (remove if you have real qarray)
# ---------------------------------------------------------------------
try:
    import qarray
except ImportError:
    class qarray:
        @staticmethod
        def set_array(a, idx, val):
            a = a.copy()
            a[idx] = val
            return a
        @staticmethod
        def gt(x, y):  return np.greater(x, y)
        @staticmethod
        def geq(x, y): return np.greater_equal(x, y)
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
    """Parallel, Bayesian, exponentially‑weighted regression with
    per‑feature decay.

    Cross‑product moments x_i x_j are forgotten using the *geometric
    mean* √(ρ_i ρ_j).  When all τ_k are equal this reduces to a single
    decay factor ρ, reproducing the legacy behaviour exactly.
    """
    # ---- validation --------------------------------------------------
    for arr in (x, y, prior_mean, prior_covar):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype.kind in ('f', 'i')
        assert not np.any(np.isinf(arr))

    assert x.ndim == 3, "`x` must be K×T×N"
    K, T, N = x.shape
    assert y.shape == (T, N)
    assert prior_mean.shape == (K,)
    assert prior_covar.shape == (K, K)
    assert qarray.is_positive_definite(prior_covar)

    if decay_scales is None:
        assert isinstance(decay_scale, Real) and qarray.gt(decay_scale, 0)
        decay_vec = np.full(K, np.exp(-1.0 / decay_scale), DT_DOUBLE)
    else:
        decay_scales = np.asarray(decay_scales, dtype=DT_DOUBLE)
        assert decay_scales.shape == (K,)
        assert np.all(qarray.gt(decay_scales, 0))
        decay_vec = np.exp(-1.0 / decay_scales)          # (K,)

    # scalar decay for target-side moments
    decay_y = np.exp(-1.0 / decay_scale) if np.isfinite(decay_scale) else 1.0

    # ---- decay matrix for X^T X using geometric mean -----------------
    if np.allclose(decay_vec, decay_vec[0]):
        # all identical → use scalar decay (matches original exactly)
        decay_outer = decay_vec[0]
    else:
        decay_outer = np.sqrt(decay_vec[:, None] * decay_vec[None, :])  # K×K

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
        xy  = swxy.T[:, None, :]          # N × 1 × K
        xx  = swx2.T                      # N × K × K
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
        # decay previous moments
        sw    *= decay_y
        swy2  *= decay_y
        swxy  *= decay_vec[:, None]                  # K × N
        if np.isscalar(decay_outer):
            swx2 *= decay_outer                      # scalar fast path
        else:
            swx2 *= decay_outer[:, :, None]          # K × K × N

        # accumulate current observation
        sw    += w[t]
        swy2  += w[t] * np.square(y0[t])
        swxy  += w[t] * y0[t] * x0[:, t]
        swx2  += w[t] * x0[:, t] * x0[:, t][:, None]

        # two-pass variance / beta refinement
        beta0      = _iterate_beta(beta_init)
        beta1      = _iterate_beta(beta0)
        beta[:, t] = beta1.T

    return beta
