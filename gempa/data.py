"""Pengambilan dan pemuatan katalog gempa wilayah Jepang.

Sumber default: USGS ComCat (FDSN web service), dibatasi kotak wilayah Jepang.
Katalog lain (mis. katalog hiposenter JMA atau NIED) bisa dipakai selama
dikonversi ke CSV dengan kolom: time, latitude, longitude, depth, mag.
"""
from __future__ import annotations

import io
import time as _time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
# Kotak wilayah Jepang + palung di sekitarnya (Kuril, Jepang, Izu-Bonin, Nankai, Ryukyu)
JAPAN_BBOX = dict(minlatitude=24.0, maxlatitude=46.0, minlongitude=122.0, maxlongitude=148.0)

_COLUMN_ALIASES = {
    "time": ["time", "origin_time", "datetime", "date", "waktu"],
    "latitude": ["latitude", "lat", "lintang"],
    "longitude": ["longitude", "lon", "lng", "bujur"],
    "depth": ["depth", "depth_km", "kedalaman"],
    "mag": ["mag", "magnitude", "m", "mj", "magnitudo"],
}


def fetch_usgs(start: str, end: str, minmag: float, out_path: str, bbox: dict | None = None) -> pd.DataFrame:
    """Unduh katalog per tahun (batas 20.000 event per permintaan FDSN) lalu simpan ke CSV."""
    bbox = bbox or JAPAN_BBOX
    years = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="YS")
    edges = [pd.Timestamp(start)] + [y for y in years if y > pd.Timestamp(start)] + [pd.Timestamp(end)]
    frames = []
    for a, b in zip(edges[:-1], edges[1:]):
        params = dict(format="csv", orderby="time-asc", minmagnitude=minmag,
                      starttime=a.strftime("%Y-%m-%d"), endtime=b.strftime("%Y-%m-%d"), **bbox)
        url = USGS_URL + "?" + urllib.parse.urlencode(params)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    text = r.read().decode("utf-8")
                break
            except Exception as e:  # jaringan putus / rate limit
                if attempt == 3:
                    raise RuntimeError(f"Gagal mengunduh {a.date()}..{b.date()}: {e}") from e
                _time.sleep(2 ** (attempt + 1))
        if text.strip():
            df = pd.read_csv(io.StringIO(text))
            frames.append(df)
            print(f"  {a.date()} .. {b.date()}: {len(df)} event")
    cat = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keep = [c for c in ["time", "latitude", "longitude", "depth", "mag", "magType", "place", "id"] if c in cat]
    cat = cat[keep].drop_duplicates(subset="id" if "id" in cat else None)
    cat.to_csv(out_path, index=False)
    return cat


def load_catalog(path: str) -> pd.DataFrame:
    """Muat CSV katalog; kembalikan DataFrame terurut waktu dengan kolom t (hari sejak epoch)."""
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for std, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = std
                break
        else:
            if std != "depth":
                raise ValueError(f"Kolom '{std}' tidak ditemukan di {path} (alias: {aliases})")
    df = df.rename(columns=rename)
    if "depth" not in df:
        df["depth"] = np.nan
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag"]).sort_values("time").reset_index(drop=True)
    df["t"] = (df["time"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() / 86400.0
    return df


def to_days(ts) -> float:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() / 86400.0


def from_days(d: float) -> pd.Timestamp:
    return pd.Timestamp("1970-01-01", tz="UTC") + pd.to_timedelta(d, unit="D")
