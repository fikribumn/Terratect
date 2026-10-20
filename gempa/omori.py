"""Hukum peluruhan Omori-Utsu: n(t) = K / (t + c)^p.

Dipakai untuk mengukur "waktu luruh" aktivitas susulan dari katalog itu sendiri.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .geology import gk_windows, haversine


def omori_integral(a, b, c, p):
    """Integral (s + c)^-p ds dari a ke b (a, b >= 0)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if abs(p - 1.0) < 1e-6:
        return np.log((b + c) / (a + c))
    return ((a + c) ** (1 - p) - (b + c) ** (1 - p)) / (p - 1)


def fit_omori_utsu(dt, t_start, t_end):
    """MLE (Ogata 1983) untuk K, c, p dari waktu susulan dt (hari setelah mainshock)."""
    dt = np.asarray(dt, float)
    dt = dt[(dt >= t_start) & (dt <= t_end)]
    n = len(dt)
    if n < 30:
        return None

    def nll(x):
        c, p = np.exp(x[0]), x[1]
        integ = omori_integral(t_start, t_end, c, p)
        K = n / integ  # K optimal secara analitik
        return -(n * np.log(K) - p * np.log(dt + c).sum() - n)

    best = None
    for c0 in (0.01, 0.1, 0.5):
        for p0 in (0.9, 1.1, 1.3):
            r = minimize(nll, [np.log(c0), p0], method="L-BFGS-B",
                         bounds=[(np.log(1e-4), np.log(5.0)), (0.5, 2.0)])
            if best is None or r.fun < best.fun:
                best = r
    c, p = float(np.exp(best.x[0])), float(best.x[1])
    K = n / omori_integral(t_start, t_end, c, p)
    return dict(K=float(K), c=c, p=p, n=int(n))


def fit_sequences(cat, mc, main_mag=6.0, duration=365.0, t_start=0.0):
    """Cari sekuens susulan untuk mainshock M>=main_mag dan fit Omori-Utsu masing-masing.

    Susulan = event M>=mc dalam radius jendela Gardner-Knopoff & `duration` hari setelah mainshock.
    """
    t, lat, lon, mag = (cat[k].to_numpy() for k in ("t", "latitude", "longitude", "mag"))
    L, _ = gk_windows(mag)
    results = []
    for i in np.where(mag >= main_mag)[0]:
        # lewati jika ada event lebih besar di sekitarnya 30 hari sebelumnya (bukan mainshock)
        lo = np.searchsorted(t, t[i] - 30)
        prev = np.arange(lo, i)
        if len(prev) and np.any((mag[prev] > mag[i]) & (haversine(lat[i], lon[i], lat[prev], lon[prev]) < L[i])):
            continue
        hi = np.searchsorted(t, t[i] + duration, side="right")
        j = np.arange(i + 1, hi)
        j = j[(mag[j] >= mc) & (haversine(lat[i], lon[i], lat[j], lon[j]) <= L[i])]
        fit = fit_omori_utsu(t[j] - t[i], t_start, duration)
        if fit:
            fit.update(mainshock_time=float(t[i]), mainshock_mag=float(mag[i]),
                       lat=float(lat[i]), lon=float(lon[i]))
            results.append(fit)
    return results


def pooled_parameters(fits, default_c=0.05, default_p=1.1, min_sequences=3):
    """Gabungkan hasil per sekuens (median berbobot jumlah event) -> c, p regional."""
    if len(fits) < min_sequences:
        return dict(c=default_c, p=default_p, n_sequences=0)
    w = np.array([f["n"] for f in fits], float)
    ps = np.array([f["p"] for f in fits])
    cs = np.array([f["c"] for f in fits])
    order = np.argsort(ps)
    p = float(ps[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)])
    order = np.argsort(cs)
    c = float(cs[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)])
    return dict(c=c, p=p, n_sequences=len(fits))
