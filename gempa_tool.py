#!/usr/bin/env python3
"""Tools prakiraan probabilistik gempa wilayah Jepang.

Contoh:
  python gempa_tool.py fetch --start 2000-01-01 --minmag 4.0 --out data/japan_usgs.csv
  python gempa_tool.py backtest --catalog data/japan_usgs.csv --out hasil/
  python gempa_tool.py predict --model hasil/model.pkl --catalog data/japan_usgs.csv --out hasil/
"""
import argparse

import pandas as pd

from gempa.features import FeatureConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="unduh katalog USGS ComCat wilayah Jepang")
    f.add_argument("--start", default="2000-01-01")
    f.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    f.add_argument("--minmag", type=float, default=4.0)
    f.add_argument("--out", default="data/japan_usgs.csv")

    def add_cfg(p):
        p.add_argument("--target-mag", type=float, default=5.0, help="magnitudo minimum yang diprakirakan")
        p.add_argument("--window-days", type=float, default=7.0, help="panjang jendela prakiraan (hari)")
        p.add_argument("--step-days", type=float, default=None, help="jarak antar waktu terbit (default = window)")
        p.add_argument("--cell-deg", type=float, default=1.0, help="ukuran sel grid (derajat)")
        p.add_argument("--radius-km", type=float, default=100.0)

    b = sub.add_parser("backtest", help="latih, kalibrasi, dan uji keandalan peluang")
    b.add_argument("--catalog", required=True)
    b.add_argument("--out", default="hasil")
    b.add_argument("--train-end", default="2016-01-01")
    b.add_argument("--cal-end", default="2020-01-01")
    b.add_argument("--split-auto", action="store_true", help="bagi 60/20/20 otomatis, abaikan tanggal")
    b.add_argument("--mc", type=float, default=None, help="magnitudo kelengkapan (default: otomatis)")
    b.add_argument("--model", choices=["gbm", "logit", "mlp"], default="gbm")
    b.add_argument("--threshold", type=float, default=0.8)
    add_cfg(b)

    p = sub.add_parser("predict", help="prakiraan peluang untuk jendela berikutnya")
    p.add_argument("--model", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", default="hasil")
    p.add_argument("--at", default=None, help="waktu terbit (UTC), default: setelah event terakhir")
    p.add_argument("--top", type=int, default=15)

    s = sub.add_parser("synthetic", help="buat katalog ETAS sintetis untuk uji kode")
    s.add_argument("--out", default="data/synthetic.csv")
    s.add_argument("--years", type=float, default=26)
    s.add_argument("--seed", type=int, default=0)

    a = ap.parse_args()
    if a.cmd == "fetch":
        import os
        from gempa.data import fetch_usgs
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        cat = fetch_usgs(a.start, a.end, a.minmag, a.out)
        print(f"Tersimpan {len(cat):,} event ke {a.out}")
    elif a.cmd == "backtest":
        from gempa.pipeline import run_backtest
        cfg = FeatureConfig(cell_deg=a.cell_deg, target_mag=a.target_mag, window_days=a.window_days,
                            step_days=a.step_days or a.window_days, radius_km=a.radius_km)
        tr, ca = (None, None) if a.split_auto else (a.train_end, a.cal_end)
        run_backtest(a.catalog, a.out, cfg, tr, ca, mc=a.mc, kind=a.model, threshold=a.threshold)
    elif a.cmd == "predict":
        from gempa.pipeline import run_predict
        run_predict(a.model, a.catalog, a.out, at=a.at, top=a.top)
    elif a.cmd == "synthetic":
        import os
        from gempa.synthetic import simulate_etas
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df = simulate_etas(years=a.years, seed=a.seed)
        df.to_csv(a.out, index=False)
        print(f"Katalog SINTETIS {len(df):,} event -> {a.out} (hanya untuk uji kode)")


if __name__ == "__main__":
    main()
