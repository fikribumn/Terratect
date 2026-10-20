"""Faktor geologi/tektonik: jarak ke palung & sesar utama, b-value, laju latar, dekluster.

Koordinat batas lempeng di bawah ini adalah APROKSIMASI (dipermudah dari peta tektonik
umum, akurasi ~10-30 km), cukup untuk fitur skala grid 0.5-1 derajat.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6371.0

# (lon, lat)
TECTONIC_LINES = {
    "kuril_trench": [(144.5, 41.2), (146.0, 42.3), (147.5, 43.5), (148.5, 44.5)],
    "japan_trench": [(144.5, 41.2), (144.0, 40.0), (143.8, 39.0), (143.5, 38.0), (142.8, 36.5),
                     (142.2, 35.0), (142.0, 34.2)],
    "izu_bonin_trench": [(142.0, 34.2), (141.9, 33.0), (142.3, 31.0), (142.5, 29.0), (142.8, 27.0),
                         (143.0, 25.0), (143.5, 24.0)],
    "sagami_trough": [(139.2, 35.1), (139.6, 34.8), (140.4, 34.4), (141.2, 34.2), (142.0, 34.2)],
    "nankai_trough": [(138.6, 35.0), (138.4, 34.4), (137.5, 33.9), (136.5, 33.4), (135.5, 33.0),
                      (134.5, 32.8), (133.5, 32.4), (132.5, 31.8), (131.8, 31.2)],
    "ryukyu_trench": [(131.8, 31.2), (131.4, 30.0), (130.5, 28.5), (129.5, 27.3), (128.2, 26.0),
                      (127.0, 25.0), (125.5, 24.2), (124.0, 23.8)],
    "japan_sea_margin": [(139.2, 46.0), (139.3, 44.0), (139.5, 42.5), (139.6, 41.0), (139.3, 39.5),
                         (138.8, 38.5), (138.3, 37.8)],
    "median_tectonic_line": [(138.2, 35.5), (137.0, 34.8), (135.5, 34.3), (134.0, 34.0), (132.5, 33.6),
                             (131.5, 33.2)],
    "itoigawa_shizuoka_tl": [(137.9, 37.0), (138.0, 36.3), (138.2, 35.6), (138.4, 35.0)],
}
TRENCHES = ["kuril_trench", "japan_trench", "izu_bonin_trench", "sagami_trough", "nankai_trough", "ryukyu_trench"]


def haversine(lat1, lon1, lat2, lon2):
    """Jarak (km); mendukung broadcasting numpy."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _densify(line, step_km=5.0):
    pts = []
    for (lo1, la1), (lo2, la2) in zip(line[:-1], line[1:]):
        n = max(2, int(haversine(la1, lo1, la2, lo2) / step_km) + 1)
        f = np.linspace(0, 1, n)
        pts.append(np.column_stack([lo1 + f * (lo2 - lo1), la1 + f * (la2 - la1)]))
    return np.vstack(pts)


_DENSE = {k: _densify(v) for k, v in TECTONIC_LINES.items()}


def distance_to_line(lat, lon, name):
    p = _DENSE[name]
    lat = np.atleast_1d(lat)[:, None]
    lon = np.atleast_1d(lon)[:, None]
    return haversine(lat, lon, p[None, :, 1], p[None, :, 0]).min(axis=1)


def tectonic_features(lat, lon):
    """Jarak ke palung terdekat, ke sesar/batas di daratan & laut Jepang, dan id palung terdekat."""
    d = {k: distance_to_line(lat, lon, k) for k in TECTONIC_LINES}
    trench_d = np.column_stack([d[k] for k in TRENCHES])
    other = np.column_stack([d[k] for k in TECTONIC_LINES if k not in TRENCHES])
    return {
        "dist_trench_km": trench_d.min(axis=1),
        "nearest_trench_id": trench_d.argmin(axis=1).astype(float),
        "dist_crustal_line_km": other.min(axis=1),
    }


def mc_maxc(mags, dm=0.1, correction=0.2):
    """Magnitudo kelengkapan (Mc) metode maximum curvature + koreksi (Woessner & Wiemer 2005)."""
    mags = np.asarray(mags)
    bins = np.arange(np.floor(mags.min() / dm) * dm, mags.max() + dm, dm)
    hist, edges = np.histogram(mags, bins=bins)
    return float(np.round(edges[hist.argmax()] + correction, 1))


def b_value_mle(mags, mc, dm=0.1):
    """b-value Aki-Utsu (MLE) dan ketidakpastian Shi & Bolt (1982)."""
    m = np.asarray(mags)
    m = m[m >= mc - 1e-9]
    n = len(m)
    if n < 2:
        return np.nan, np.nan
    mean = m.mean()
    b = np.log10(np.e) / (mean - (mc - dm / 2))
    sb = 2.3 * b ** 2 * np.sqrt(((m - mean) ** 2).sum() / (n * (n - 1)))
    return float(b), float(sb)


def gk_windows(m):
    """Jendela ruang (km) & waktu (hari) Gardner & Knopoff (1974)."""
    m = np.asarray(m, float)
    L = 10 ** (0.1238 * m + 0.983)
    T = np.where(m >= 6.5, 10 ** (0.032 * m + 2.7389), 10 ** (0.5409 * m - 0.547))
    return L, T


def decluster_gk(t, lat, lon, mag):
    """Kembalikan mask boolean event 'mainshock/latar' (True) setelah dekluster Gardner-Knopoff."""
    t, lat, lon, mag = map(np.asarray, (t, lat, lon, mag))
    n = len(t)
    is_main = np.ones(n, bool)
    L, T = gk_windows(mag)
    for i in np.argsort(-mag, kind="stable"):
        if not is_main[i]:
            continue
        lo, hi = np.searchsorted(t, t[i] - T[i]), np.searchsorted(t, t[i] + T[i], side="right")
        j = np.arange(lo, hi)
        j = j[(j != i) & is_main[j] & (mag[j] <= mag[i])]
        if len(j):
            near = haversine(lat[i], lon[i], lat[j], lon[j]) <= L[i]
            is_main[j[near]] = False
    return is_main
