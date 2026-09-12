"""Implied-volatility smoothers with a uniform interface, for head-to-head
comparison. Standalone (reusable for a separate methods paper) -- depends only
on numpy/scipy/statsmodels, not on the kalshi_arb package.

Every smoother fits IV(strike) on one day's OTM-combined smile and exposes the
same call signature, so `compare_smoothers.py` can swap them freely:

    sm = LSQSpline().fit(K, iv, F, T)
    iv_hat = sm(K_grid)

`rnd_from_iv` turns any IV(K) into a risk-neutral density with ONE shared
pipeline, so a comparison isolates the smoother -- everything downstream is
identical. It returns both the RAW density (only tiny negatives clipped) and the
ARBITRAGE-REPAIRED density (isotonic + light smoothing, as in production); the
raw one reveals how much each method leans on the repair.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, PchipInterpolator
from scipy.optimize import least_squares
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _bs_call(S, K, T, r, sigma):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _isotonic_increasing(y):
    y = np.asarray(y, float)
    vals, cnts = [], []
    for yi in y:
        vals.append(yi); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            nc = cnts[-1] + cnts[-2]
            nv = (vals[-1] * cnts[-1] + vals[-2] * cnts[-2]) / nc
            vals.pop(); cnts.pop(); vals[-1] = nv; cnts[-1] = nc
    out = np.empty(len(y)); i = 0
    for v, c in zip(vals, cnts):
        out[i:i + c] = v; i += c
    return out


def rnd_from_iv(grid, sigma, S, T, r, smooth_window=25):
    """IV(grid) -> (raw_density, repaired_density) via Breeden-Litzenberger.

    raw = e^{rT} d2C/dK2 with only negatives clipped (shows the method's native
    smoothness / arbitrage). repaired = production pipeline (clip dC/dK to
    [-DF,0], isotonic, clip, light smooth) -- always non-negative.
    """
    calls = _bs_call(S, grid, T, r, sigma)
    DF = np.exp(-r * T)
    d2 = np.gradient(np.gradient(calls, grid), grid)
    raw = np.clip(np.exp(r * T) * d2, 0.0, None)
    fp = _isotonic_increasing(np.clip(np.gradient(calls, grid), -DF, 0.0))
    rep = np.clip(np.exp(r * T) * np.gradient(fp, grid), 0.0, None)
    if smooth_window and smooth_window > 1:
        rep = np.convolve(rep, np.ones(smooth_window) / smooth_window, mode="same")
    # also return the pre-clip signed 2nd derivative to measure arbitrage
    signed = np.exp(r * T) * d2
    return raw[2:-2], rep[2:-2], signed[2:-2], grid[2:-2]


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class Smoother:
    name = "base"
    def fit(self, K, iv, F, T):
        self.K = np.asarray(K, float); self.iv = np.asarray(iv, float)
        self.F = float(F); self.T = float(T)
        self.k_min, self.k_max = self.K.min(), self.K.max()
        self.iv_min, self.iv_max = self.iv.min(), self.iv.max()
        self._fit()
        return self
    def _fit(self): ...
    def _raw(self, Kq): raise NotImplementedError
    def __call__(self, Kq, clamp_wings=True):
        Kq = np.asarray(Kq, float)
        x = np.clip(Kq, self.k_min, self.k_max) if clamp_wings else Kq
        return self._raw(x)
    ok = True


# --------------------------------------------------------------------------- #
# 1. LSQ spline  (the production method: OTM, quantile knots, IV-envelope clamp)
# --------------------------------------------------------------------------- #
class LSQSpline(Smoother):
    name = "LSQ spline (production)"
    def __init__(self, n_knots=6): self.n_knots = n_knots
    def _fit(self):
        K, iv = np.unique(self.K, return_index=True)
        iv = self.iv[iv]
        self._sp = None
        for k in range(min(self.n_knots, len(K) - 2), 0, -1):
            interior = np.unique(np.quantile(K, np.linspace(0, 1, k + 2)[1:-1]))
            interior = interior[(interior > K.min()) & (interior < K.max())]
            if len(interior) == 0:
                continue
            try:
                self._sp = LSQUnivariateSpline(K, iv, interior); break
            except ValueError:
                continue
        self.ok = self._sp is not None
    def _raw(self, Kq):
        return np.clip(self._sp(Kq), self.iv_min, self.iv_max)


# --------------------------------------------------------------------------- #
# 2. cubic smoothing spline (scipy UnivariateSpline, s tuned to noise)
# --------------------------------------------------------------------------- #
class CubicSpline(Smoother):
    name = "cubic smoothing spline"
    def __init__(self, s_frac=0.5): self.s_frac = s_frac
    def _fit(self):
        K, idx = np.unique(self.K, return_index=True); iv = self.iv[idx]
        s = self.s_frac * len(K) * np.var(iv)
        self._sp = UnivariateSpline(K, iv, k=3, s=max(s, 1e-8))
    def _raw(self, Kq): return self._sp(Kq)


# --------------------------------------------------------------------------- #
# 3. polynomial (deg 4) in log-moneyness
# --------------------------------------------------------------------------- #
class Polyfit(Smoother):
    name = "polynomial (deg 4)"
    def __init__(self, deg=4): self.deg = deg
    def _fit(self):
        self._c = np.polyfit(np.log(self.K / self.F), self.iv, self.deg)
    def _raw(self, Kq):
        return np.clip(np.polyval(self._c, np.log(Kq / self.F)), 1e-3, None)


# --------------------------------------------------------------------------- #
# 4. PCHIP (monotone, shape-preserving interpolation)
# --------------------------------------------------------------------------- #
class Pchip(Smoother):
    name = "PCHIP (monotone)"
    def _fit(self):
        K, idx = np.unique(self.K, return_index=True)
        self._p = PchipInterpolator(K, self.iv[idx], extrapolate=True)
    def _raw(self, Kq): return self._p(Kq)


# --------------------------------------------------------------------------- #
# 5. LOWESS (local regression) + linear interp to the grid
# --------------------------------------------------------------------------- #
class Lowess(Smoother):
    name = "LOWESS"
    def __init__(self, frac=0.3): self.frac = frac
    def _fit(self):
        sm = lowess(self.iv, self.K, frac=self.frac, return_sorted=True)
        self._x, self._y = sm[:, 0], sm[:, 1]
    def _raw(self, Kq): return np.interp(Kq, self._x, self._y)


# --------------------------------------------------------------------------- #
# 6. SABR (Hagan 2002 lognormal; beta fixed, calibrate alpha, rho, nu)
# --------------------------------------------------------------------------- #
def _sabr_vol(K, F, T, alpha, beta, rho, nu):
    K = np.asarray(K, float)
    eps = 1e-9
    logFK = np.log(F / K)
    FK = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK * logFK
    disc = np.clip(1 - 2 * rho * z + z ** 2, 1e-12, None)   # guard sqrt
    xz = np.log((np.sqrt(disc) + z - rho) / (1 - rho))
    zx = np.where(np.abs(z) < eps, 1.0, z / np.where(np.abs(xz) < eps, eps, xz))
    A = alpha / (FK * (1 + ((1 - beta) ** 2 / 24) * logFK ** 2 +
                       ((1 - beta) ** 4 / 1920) * logFK ** 4))
    C = 1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / FK ** 2 +
             0.25 * rho * beta * nu * alpha / FK +
             (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    return A * zx * C


class SABR(Smoother):
    name = "SABR (Hagan, β=0.5)"
    def __init__(self, beta=0.5): self.beta = beta
    def _fit(self):
        F, T, b = self.F, self.T, self.beta
        atm = np.interp(F, self.K, self.iv)
        # for beta<1 the ATM relation is atm ~ alpha / F^(1-beta), so alpha scales
        # like atm*F^(1-beta) (~9 for SPX, beta=0.5): bound generously around it.
        a0 = max(atm * F ** (1 - b), 1e-3)
        a_hi = max(50.0, 10 * a0)
        def resid(p):
            a, rho, nu = p
            v = _sabr_vol(self.K, F, T, a, b, rho, nu)
            return np.nan_to_num(v - self.iv, nan=1e2, posinf=1e2, neginf=1e2)
        try:
            r = least_squares(resid, [a0, -0.3, 0.5],
                              bounds=([1e-4, -0.999, 1e-4], [a_hi, 0.999, 20.0]),
                              max_nfev=6000)
            self._p = r.x
            self.ok = r.success and np.isfinite(r.cost)
        except Exception:
            self.ok = False
    def _raw(self, Kq):
        a, rho, nu = self._p
        return _sabr_vol(Kq, self.F, self.T, a, self.beta, rho, nu)


# --------------------------------------------------------------------------- #
# 7. SVI (Gatheral raw parameterization of total variance)
# --------------------------------------------------------------------------- #
class SVI(Smoother):
    name = "SVI (Gatheral raw)"
    def _fit(self):
        k = np.log(self.K / self.F)
        w = (self.iv ** 2) * self.T                     # total variance
        a0 = max(w.min(), 1e-6)
        p0 = [a0, 0.1, -0.3, 0.0, 0.1]                  # a, b, rho, m, sigma
        def resid(p):
            a, b, rho, m, sig = p
            wm = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))
            return wm - w
        try:
            r = least_squares(resid, p0,
                              bounds=([-1.0, 0.0, -0.999, -2.0, 1e-4],
                                      [5.0, 10.0, 0.999, 2.0, 5.0]), max_nfev=4000)
            self._p = r.x; self.ok = r.success or r.cost < 1e2
        except Exception:
            self.ok = False
    def _raw(self, Kq):
        a, b, rho, m, sig = self._p
        k = np.log(Kq / self.F)
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))
        return np.sqrt(np.clip(w, 1e-8, None) / self.T)


ALL_SMOOTHERS = [LSQSpline, CubicSpline, Polyfit, Pchip, Lowess, SABR, SVI]
