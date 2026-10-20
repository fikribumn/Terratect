"""Katalog sintetis ETAS — HANYA untuk menguji kode, bukan data Jepang sungguhan."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import rupture_length_km
from .geology import TECTONIC_LINES, TRENCHES, _densify


def simulate_etas(years=26.0, mu_per_year=400.0, mc=4.0, b=1.0, mmax=9.0, K=0.12, alpha=0.9,
                  c=0.02, p=1.15, seed=0, start="2000-01-01"):
    rng = np.random.default_rng(seed)
    T = years * 365.25
    pts = np.vstack([_densify(TECTONIC_LINES[k]) for k in TRENCHES])

    def gr(n):
        u = rng.random(n)
        beta = b * np.log(10)
        return mc - np.log(1 - u * (1 - np.exp(-beta * (mmax - mc)))) / beta

    n0 = rng.poisson(mu_per_year * years)
    src = pts[rng.integers(len(pts), size=n0)]
    gen = dict(t=rng.uniform(0, T, n0), lon=src[:, 0] + rng.normal(0, 0.4, n0),
               lat=src[:, 1] + rng.normal(0, 0.3, n0), mag=gr(n0))
    events = [gen]
    while len(gen["t"]):
        nk = rng.poisson(K * 10 ** (alpha * (gen["mag"] - mc)))
        if nk.sum() == 0:
            break
        par = np.repeat(np.arange(len(nk)), nk)
        u = rng.random(len(par))
        dt = c * ((1 - u) ** (-1 / (p - 1)) - 1)
        sig = np.maximum(10.0, 0.5 * rupture_length_km(gen["mag"][par])) / 111.0
        ang = rng.uniform(0, 2 * np.pi, len(par))
        r = np.abs(rng.normal(0, sig))
        gen = dict(t=gen["t"][par] + dt, lon=gen["lon"][par] + r * np.cos(ang),
                   lat=gen["lat"][par] + r * np.sin(ang), mag=gr(len(par)))
        keep = gen["t"] < T
        gen = {k: v[keep] for k, v in gen.items()}
        events.append(gen)
    ev = {k: np.concatenate([e[k] for e in events]) for k in ("t", "lon", "lat", "mag")}
    df = pd.DataFrame({
        "time": pd.Timestamp(start, tz="UTC") + pd.to_timedelta(ev["t"], unit="D"),
        "latitude": ev["lat"], "longitude": ev["lon"],
        "depth": rng.uniform(5, 80, len(ev["t"])), "mag": np.round(ev["mag"], 1),
    })
    df = df[(df.latitude.between(24, 46)) & (df.longitude.between(122, 148))]
    return df.sort_values("time").reset_index(drop=True)
