"""Pembentukan fitur per (waktu terbit prakiraan, sel grid).

Fitur dinamis : ekspektasi susulan Omori-Utsu (waktu luruh), jumlah event terbaru,
                magnitudo maksimum terbaru, waktu sejak gempa besar terakhir.
Fitur statis  : laju latar (terdekluster), b-value lokal, kedalaman, jarak ke palung/sesar.
Label         : 1 jika ada gempa M >= target_mag di sel tsb dalam [t, t + window_days).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import sparse

from .data import JAPAN_BBOX
from .geology import b_value_mle, decluster_gk, haversine, tectonic_features
from .omori import omori_integral


@dataclass
class FeatureConfig:
    cell_deg: float = 1.0
    target_mag: float = 5.0
    window_days: float = 7.0
    step_days: float = 7.0
    radius_km: float = 100.0
    alpha: float = 1.0          # produktivitas susulan ~ 10^(alpha (M - Mc))
    sigma_min_km: float = 20.0  # lebar minimum kernel spasial
    min_cell_events: int = 5    # sel aktif = punya >= n event (M>=Mc) di periode latih

    def to_dict(self):
        return asdict(self)


def rupture_length_km(m):
    """Panjang rupture kasar (Wells & Coppersmith 1994 disederhanakan)."""
    return 10 ** (0.5 * np.asarray(m, float) - 1.85)


class Grid:
    def __init__(self, cfg: FeatureConfig, train_cat: pd.DataFrame, bbox: dict | None = None):
        bbox = bbox or JAPAN_BBOX
        self.cfg = cfg
        self.lat0, self.lon0 = bbox["minlatitude"], bbox["minlongitude"]
        self.nlat = int(np.ceil((bbox["maxlatitude"] - self.lat0) / cfg.cell_deg))
        self.nlon = int(np.ceil((bbox["maxlongitude"] - self.lon0) / cfg.cell_deg))
        raw = self._raw_index(train_cat["latitude"].to_numpy(), train_cat["longitude"].to_numpy())
        counts = np.bincount(raw[raw >= 0], minlength=self.nlat * self.nlon)
        self.active_raw = np.where(counts >= cfg.min_cell_events)[0]
        self.lookup = -np.ones(self.nlat * self.nlon, int)
        self.lookup[self.active_raw] = np.arange(len(self.active_raw))
        i, j = np.divmod(self.active_raw, self.nlon)
        self.lat = self.lat0 + (i + 0.5) * cfg.cell_deg
        self.lon = self.lon0 + (j + 0.5) * cfg.cell_deg

    def _raw_index(self, lat, lon):
        i = np.floor((lat - self.lat0) / self.cfg.cell_deg).astype(int)
        j = np.floor((lon - self.lon0) / self.cfg.cell_deg).astype(int)
        ok = (i >= 0) & (i < self.nlat) & (j >= 0) & (j < self.nlon)
        return np.where(ok, i * self.nlon + j, -1)

    def cell_of(self, lat, lon):
        raw = self._raw_index(np.asarray(lat), np.asarray(lon))
        return np.where(raw >= 0, self.lookup[np.maximum(raw, 0)], -1)

    def __len__(self):
        return len(self.active_raw)


def static_features(grid: Grid, train_cat: pd.DataFrame, mc: float, global_b: float) -> pd.DataFrame:
    """Fitur 'keadaan geologi' per sel, dihitung HANYA dari periode latih (tanpa kebocoran)."""
    cfg = grid.cfg
    t, lat, lon, mag = (train_cat[k].to_numpy() for k in ("t", "latitude", "longitude", "mag"))
    depth = train_cat["depth"].to_numpy(float)
    years = max((t.max() - t.min()) / 365.25, 1e-3)
    main = decluster_gk(t, lat, lon, mag)
    d = haversine(grid.lat[:, None], grid.lon[:, None], lat[None, :], lon[None, :])  # sel x event
    near50, near100 = d <= 50.0, d <= 100.0

    bg_rate = (near50 & main[None, :]).sum(1) / years
    cell = grid.cell_of(lat, lon)
    tgt = np.bincount(cell[(cell >= 0) & (mag >= cfg.target_mag)], minlength=len(grid)) / years

    b_local = np.full(len(grid), global_b)
    for k in range(len(grid)):
        m = mag[near100[k]]
        if (m >= mc).sum() >= 50:
            b, _ = b_value_mle(m, mc)
            b_local[k] = np.clip(b, 0.5, 2.0)
    dep = np.where(near50, depth[None, :], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # sel tanpa event -> NaN, diisi median
        mean_depth = np.nanmean(dep, axis=1)
    frac_deep = np.nansum(dep > 70, axis=1) / np.maximum(near50.sum(1), 1)
    mean_depth = np.where(np.isnan(mean_depth), np.nanmedian(depth), mean_depth)

    out = pd.DataFrame({
        "log_bg_rate": np.log10(bg_rate + 1e-2),
        "log_target_rate": np.log10(tgt + 1e-3),
        "b_value": b_local,
        "mean_depth_km": mean_depth,
        "frac_deep": frac_deep,
        "cell_lat": grid.lat,
        "cell_lon": grid.lon,
    })
    for k, v in tectonic_features(grid.lat, grid.lon).items():
        out[k] = v
    return out


class DynamicBuilder:
    """Hitung fitur dinamis secara efisien memakai matriks bobot spasial sparse."""

    def __init__(self, cat: pd.DataFrame, grid: Grid, cfg: FeatureConfig, mc: float, c: float, p: float):
        self.cfg, self.grid, self.c, self.p = cfg, grid, c, p
        self.t = cat["t"].to_numpy()
        self.mag = cat["mag"].to_numpy()
        lat, lon = cat["latitude"].to_numpy(), cat["longitude"].to_numpy()
        self.prod = 10 ** (cfg.alpha * (self.mag - mc))
        self.cell = grid.cell_of(lat, lon)
        sigma = np.maximum(cfg.sigma_min_km, 0.5 * rupture_length_km(self.mag))
        rows_w, cols_w, vals_w, rows_b, cols_b = [], [], [], [], []
        for s in range(0, len(self.t), 4000):
            sl = slice(s, s + 4000)
            d = haversine(lat[sl, None], lon[sl, None], grid.lat[None, :], grid.lon[None, :])
            w = np.exp(-0.5 * (d / sigma[sl, None]) ** 2)
            r, cc = np.nonzero(w > 1e-3)
            rows_w.append(r + s); cols_w.append(cc); vals_w.append(w[r, cc])
            r, cc = np.nonzero(d <= cfg.radius_km)
            rows_b.append(r + s); cols_b.append(cc)
        n, m = len(self.t), len(grid)
        cat_ = lambda xs: np.concatenate(xs) if xs else np.array([], int)
        self.W = sparse.csr_matrix((cat_(vals_w), (cat_(rows_w), cat_(cols_w))), shape=(n, m))
        self.B = sparse.csr_matrix((np.ones(len(cat_(rows_b))), (cat_(rows_b), cat_(cols_b))), shape=(n, m))
        self.WT, self.BT = self.W.T.tocsr(), self.B.T.tocsr()

    def build(self, issue_times, with_labels=True) -> pd.DataFrame:
        cfg, t, mag = self.cfg, self.t, self.mag
        T = cfg.window_days
        ncell = len(self.grid)
        p_var = {"omori": self.p, "omori_fast": self.p + 0.3, "omori_slow": max(self.p - 0.3, 0.5)}
        last_big = np.full(ncell, -np.inf)
        big_idx = np.where(mag >= cfg.target_mag)[0]
        bptr = 0
        blocks = []
        for it in np.sort(np.asarray(issue_times, float)):
            k = np.searchsorted(t, it)  # event dengan t < it
            feats = {}
            d0 = it - t[:k]
            for name, pv in p_var.items():
                g = np.zeros(len(t))
                g[:k] = self.prod[:k] * omori_integral(d0, d0 + T, self.c, pv)
                feats["log_" + name] = np.log10(self.WT @ g + 1e-6)
            for days in (1, 7, 30, 365):
                lo = np.searchsorted(t, it - days)
                ind = np.zeros(len(t))
                ind[lo:k] = 1.0
                feats[f"n_{days}d"] = self.BT @ ind
            lo = np.searchsorted(t, it - 30)
            if k > lo:
                sub = self.B[lo:k].multiply(mag[lo:k, None]).tocsc()
                feats["max_mag_30d"] = np.asarray(sub.max(axis=0).todense()).ravel()
            else:
                feats["max_mag_30d"] = np.zeros(ncell)
            while bptr < len(big_idx) and t[big_idx[bptr]] < it:
                cols = self.B[big_idx[bptr]].indices
                last_big[cols] = t[big_idx[bptr]]
                bptr += 1
            feats["log_days_since_big"] = np.log10(np.clip(it - last_big, 0.01, 3650))
            df = pd.DataFrame(feats)
            df["cell"] = np.arange(ncell)
            df["issue_t"] = it
            if with_labels:
                hi = np.searchsorted(t, it + T)
                sel = np.arange(k, hi)
                sel = sel[(mag[sel] >= cfg.target_mag) & (self.cell[sel] >= 0)]
                y = np.zeros(ncell, int)
                y[self.cell[sel]] = 1
                df["y"] = y
            blocks.append(df)
        return pd.concat(blocks, ignore_index=True)


def assemble(dyn: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    return dyn.merge(static, left_on="cell", right_index=True, how="left")


FEATURE_COLUMNS = [
    "log_omori", "log_omori_fast", "log_omori_slow",
    "n_1d", "n_7d", "n_30d", "n_365d", "max_mag_30d", "log_days_since_big",
    "log_bg_rate", "log_target_rate", "b_value", "mean_depth_km", "frac_deep",
    "dist_trench_km", "nearest_trench_id", "dist_crustal_line_km",
]
