"""IV smile functional forms, with a uniform interface and a registry.

One place defines every smoother; production selects one via `config.SMILE_METHOD`
and the comparison framework (`analysis/compare_smoothers.py`) reuses the same
classes -- no duplication, nothing deprecated. Each smoother fits IV(strike) on a
single day's OTM-combined smile and evaluates at arbitrary strikes:

    sm = fit_smile(K, iv, F, T, "sabr")   # None if calibration fails
    iv_hat = sm(K_grid)                    # strikes auto-clamped to observed range

Envelope/clamping and the OTM selection live in `smoothing.py`, which is the
pipeline-facing layer; this module is pure smile mathematics.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, PchipInterpolator
from scipy.optimize import least_squares
from statsmodels.nonparametric.smoothers_lowess import lowess
from .. import config


def fit_iv_spline(strikes, iv, n_knots=None):
    """LSQ spline of IV vs strike, interior knots at strike quantiles (guarantee
    Schoenberg-Whitney), knot count reduced + retried on sparse data. None if <4."""
    n_knots = config.SMILE_KNOTS if n_knots is None else n_knots
    strikes = np.asarray(strikes, float); iv = np.asarray(iv, float)
    order = np.argsort(strikes); strikes, iv = strikes[order], iv[order]
    strikes, idx = np.unique(strikes, return_index=True); iv = iv[idx]
    if len(strikes) < 4:
        return None
    for k in range(min(n_knots, len(strikes) - 2), 0, -1):
        interior = np.unique(np.quantile(strikes, np.linspace(0, 1, k + 2)[1:-1]))
        interior = interior[(interior > strikes.min()) & (interior < strikes.max())]
        if len(interior) == 0:
            continue
        try:
            return LSQUnivariateSpline(strikes, iv, interior)
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
class Smoother:
    name = "base"
    def fit(self, K, iv, F, T):
        self.K = np.asarray(K, float); self.iv = np.asarray(iv, float)
        self.F = float(F); self.T = float(T)
        self.k_min, self.k_max = float(self.K.min()), float(self.K.max())
        self.iv_min, self.iv_max = float(self.iv.min()), float(self.iv.max())
        self.ok = True
        self._fit()
        return self
    def _fit(self): ...
    def _raw(self, Kq): raise NotImplementedError
    def __call__(self, Kq, clamp_wings=True):
        Kq = np.asarray(Kq, float)
        x = np.clip(Kq, self.k_min, self.k_max) if clamp_wings else Kq
        return self._raw(x)


class LSQSpline(Smoother):
    name = "lsq"
    def __init__(self, n_knots=None): self.n_knots = n_knots
    def _fit(self):
        self._sp = fit_iv_spline(self.K, self.iv, self.n_knots)
        self.ok = self._sp is not None
    def _raw(self, Kq): return self._sp(Kq)


class CubicSpline(Smoother):
    name = "cubic"
    def __init__(self, s_frac=0.5): self.s_frac = s_frac
    def _fit(self):
        K, idx = np.unique(self.K, return_index=True); iv = self.iv[idx]
        self._sp = UnivariateSpline(K, iv, k=3,
                                    s=max(self.s_frac * len(K) * np.var(iv), 1e-8))
    def _raw(self, Kq): return self._sp(Kq)


class Polyfit(Smoother):
    name = "poly"
    def __init__(self, deg=4): self.deg = deg
    def _fit(self): self._c = np.polyfit(np.log(self.K / self.F), self.iv, self.deg)
    def _raw(self, Kq):
        return np.clip(np.polyval(self._c, np.log(Kq / self.F)), 1e-3, None)


class Pchip(Smoother):
    name = "pchip"
    def _fit(self):
        K, idx = np.unique(self.K, return_index=True)
        self._p = PchipInterpolator(K, self.iv[idx], extrapolate=True)
    def _raw(self, Kq): return self._p(Kq)


class Lowess(Smoother):
    name = "lowess"
    def __init__(self, frac=0.3): self.frac = frac
    def _fit(self):
        sm = lowess(self.iv, self.K, frac=self.frac, return_sorted=True)
        self._x, self._y = sm[:, 0], sm[:, 1]
    def _raw(self, Kq): return np.interp(Kq, self._x, self._y)


def _sabr_vol(K, F, T, alpha, beta, rho, nu):
    K = np.asarray(K, float); eps = 1e-9
    logFK = np.log(F / K); FK = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK * logFK
    disc = np.clip(1 - 2 * rho * z + z ** 2, 1e-12, None)
    xz = np.log((np.sqrt(disc) + z - rho) / (1 - rho))
    zx = np.where(np.abs(z) < eps, 1.0, z / np.where(np.abs(xz) < eps, eps, xz))
    A = alpha / (FK * (1 + ((1 - beta) ** 2 / 24) * logFK ** 2 +
                       ((1 - beta) ** 4 / 1920) * logFK ** 4))
    C = 1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / FK ** 2 +
             0.25 * rho * beta * nu * alpha / FK +
             (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    return A * zx * C


class SABR(Smoother):
    name = "sabr"
    def __init__(self, beta=0.5): self.beta = beta
    def _fit(self):
        F, T, b = self.F, self.T, self.beta
        a0 = max(np.interp(F, self.K, self.iv) * F ** (1 - b), 1e-3)
        def resid(p):
            a, rho, nu = p
            return np.nan_to_num(_sabr_vol(self.K, F, T, a, b, rho, nu) - self.iv,
                                 nan=1e2, posinf=1e2, neginf=1e2)
        try:
            r = least_squares(resid, [a0, -0.3, 0.5],
                              bounds=([1e-4, -0.999, 1e-4],
                                      [max(50.0, 10 * a0), 0.999, 20.0]), max_nfev=6000)
            self._p = r.x; self.ok = r.success and np.isfinite(r.cost)
        except Exception:
            self.ok = False
    def _raw(self, Kq):
        a, rho, nu = self._p
        return _sabr_vol(Kq, self.F, self.T, a, self.beta, rho, nu)


class SVI(Smoother):
    name = "svi"
    def _fit(self):
        k = np.log(self.K / self.F); w = (self.iv ** 2) * self.T
        def resid(p):
            a, b, rho, m, sig = p
            return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2)) - w
        try:
            r = least_squares(resid, [max(w.min(), 1e-6), 0.1, -0.3, 0.0, 0.1],
                              bounds=([-1.0, 0.0, -0.999, -2.0, 1e-4],
                                      [5.0, 10.0, 0.999, 2.0, 5.0]), max_nfev=6000)
            self._p = r.x; self.ok = r.success and np.isfinite(r.cost)
        except Exception:
            self.ok = False
    def _raw(self, Kq):
        a, b, rho, m, sig = self._p
        k = np.log(Kq / self.F)
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))
        return np.sqrt(np.clip(w, 1e-8, None) / self.T)


REGISTRY = {c.name: c for c in [LSQSpline, CubicSpline, Polyfit, Pchip, Lowess, SABR, SVI]}


def fit_smile(K, iv, F, T, method=None):
    """Fit the named smoother (default config.SMILE_METHOD). None if it fails."""
    method = config.SMILE_METHOD if method is None else method
    sm = REGISTRY[method]().fit(K, iv, F, T)
    return sm if sm.ok else None
