#!/usr/bin/env python3
"""Rangkaian eksperimen ML pada katalog JMA 1926-2007 (M>=4.5, kedalaman < 100 km).

  python experiments.py models     # A: perbandingan model + metode kalibrasi
  python experiments.py ablation   # B: kelompok fitur (waktu luruh vs geologi)
  python experiments.py targets    # C: definisi target (magnitudo, jendela, ukuran sel)
  python experiments.py rolling    # D: uji bergulir di 4 periode berbeda
Hasil: hasil/eksperimen/<nama>.json (+ .png)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from gempa.data import from_days, load_catalog
from gempa.evaluate import evaluate
from gempa.features import FEATURE_COLUMNS, FeatureConfig
from gempa.geology import haversine
from gempa.model import CalibratedModel, Climatology, OOFCalibratedModel
from gempa.pipeline import build_dataset

CATALOG = "data/jma_1926_2007_M4.5.csv"
OUT = "hasil/eksperimen"
CACHE = "hasil/cache"
MC = 4.5
NEG_FRAC = 0.2  # latih dengan 20% sampel negatif (bobot 5x) -> ~5x lebih cepat
SPLIT = dict(train_end="1985-01-01", cal_end="1995-01-01")

DECAY = ["log_omori", "log_omori_fast", "log_omori_slow", "log_days_since_big"]
ACTIVITY = ["n_1d", "n_7d", "n_30d", "n_365d", "max_mag_30d"]
GEOLOGY = ["log_bg_rate", "log_target_rate", "b_value", "mean_depth_km", "frac_deep",
           "dist_trench_km", "nearest_trench_id", "dist_crustal_line_km"]
FEATURE_SETS = {
    "geologi_saja": GEOLOGY,
    "waktu_luruh_saja": DECAY,
    "waktu_luruh+aktivitas": DECAY + ACTIVITY,
    "waktu_luruh+geologi": DECAY + GEOLOGY,
    "semua": FEATURE_COLUMNS,
}


def dataset(cfg: FeatureConfig, split=SPLIT, tag=None):
    """Bangun (atau ambil dari cache) dataset fitur untuk satu konfigurasi."""
    os.makedirs(CACHE, exist_ok=True)
    tag = tag or (f"M{cfg.target_mag}_w{cfg.window_days:g}_c{cfg.cell_deg:g}_s{cfg.step_days:g}"
                  f"_{split['train_end'][:4]}_{split['cal_end'][:4]}" + (f"_{split['test_end'][:4]}" if split.get("test_end") else ""))
    path = os.path.join(CACHE, tag + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    ds = build_dataset(CATALOG, cfg, train_end=split["train_end"], cal_end=split["cal_end"], mc=MC,
                       end_date=split.get("test_end"), verbose=False)
    ds.pop("cat")
    with open(path, "wb") as f:
        pickle.dump(ds, f)
    return ds


def alarm_episodes(rows, days=30.0, km=150.0):
    """Jumlah episode independen: alarm yang berdekatan (waktu & jarak) dihitung satu."""
    rows = rows.sort_values("issue_t")
    eps = []  # (t_last, lat, lon, hits, n)
    for r in rows.itertuples():
        for e in eps:
            if r.issue_t - e[0] <= days and haversine(r.cell_lat, r.cell_lon, e[1], e[2]) <= km:
                e[0], e[3], e[4] = r.issue_t, e[3] + r.y, e[4] + 1
                break
        else:
            eps.append([r.issue_t, r.cell_lat, r.cell_lon, r.y, 1])
    return dict(episodes=len(eps), episodes_with_hit=int(sum(e[3] > 0 for e in eps)),
                episode_precision=[round(e[3] / e[4], 3) for e in eps],
                episode_where=[f"{from_days(e[0]):%Y-%m} ({e[1]:.1f},{e[2]:.1f})" for e in eps])


def score(p, te, p_ref, threshold=0.8):
    r = evaluate(p, te["y"].to_numpy(), p_ref=p_ref, threshold=threshold)
    a = te.assign(p=p)
    r.update(alarm_episodes(a[a["p"] >= threshold]))
    return r


def fit_predict(kind, calib, cols, ds):
    tr, ca, te = ds["train"], ds["cal"], ds["test"]
    if calib == "isotonik_periode":
        m = CalibratedModel(kind, neg_frac=NEG_FRAC).fit(tr[cols], tr["y"], ca[cols], ca["y"])
    elif calib == "isotonik_oof":
        full = pd.concat([tr, ca])
        m = OOFCalibratedModel(kind, neg_frac=NEG_FRAC).fit_oof(full[cols], full["y"], full["issue_t"])
    elif calib == "tanpa":
        m = CalibratedModel(kind, neg_frac=NEG_FRAC)
        full = pd.concat([tr, ca])
        m._fit_base(m.base, full[cols], full["y"])
        return m.predict_raw(te[cols])
    return m.predict_proba(te[cols])


def climatology(ds):
    full = pd.concat([ds["train"], ds["cal"]])
    return Climatology().fit(full["cell"], full["y"], len(ds["grid"])).predict_proba(ds["test"]["cell"])


def summarize(r):
    keys = ["auc", "brier_skill_score", "log_loss", "alarms", "alarm_hits", "precision_at_threshold",
            "precision_ci95", "recall_at_threshold", "episodes", "episodes_with_hit"]
    return {k: r.get(k) for k in keys}


def print_table(rows, title):
    print(f"\n### {title}")
    print(f"{'nama':<38}{'AUC':>6}{'BSS':>8}{'alarm':>7}{'presisi':>9}{'CI95':>14}{'recall':>8}{'episode':>9}")
    for name, r in rows.items():
        ci = r["precision_ci95"]
        cis = f"{ci[0]:.0%}-{ci[1]:.0%}" if ci[0] == ci[0] else "-"
        prec = f"{r['precision_at_threshold']:.0%}" if r["alarms"] else "-"
        print(f"{name:<38}{r['auc']:6.3f}{r.get('brier_skill_score', 0):+8.3f}{r['alarms']:7d}{prec:>9}{cis:>14}"
              f"{r['recall_at_threshold']:8.1%}{r['episodes']:>5d}/{r['episodes_with_hit']:<3d}")


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=2, default=float)


BASE_CFG = dict(target_mag=5.0, window_days=7.0, step_days=2.0, cell_deg=1.0)


def exp_models():
    ds = dataset(FeatureConfig(**BASE_CFG))
    te, ref = ds["test"], climatology(ds)
    rows, full = {"klimatologi": score(ref, te, ref)}, {}
    physics = ["log_omori", "log_target_rate"]  # setara Reasenberg-Jones/ETAS sederhana
    runs = [("fisika_omori(logit)", "logit", "isotonik_periode", physics),
            ("fisika_omori(logit)+oof", "logit", "isotonik_oof", physics),
            ("logistik", "logit", "isotonik_periode", FEATURE_COLUMNS),
            ("logistik+oof", "logit", "isotonik_oof", FEATURE_COLUMNS),
            ("gbm_tanpa_kalibrasi", "gbm", "tanpa", FEATURE_COLUMNS),
            ("gbm", "gbm", "isotonik_periode", FEATURE_COLUMNS),
            ("gbm+oof", "gbm", "isotonik_oof", FEATURE_COLUMNS),
            ("neural_net_mlp", "mlp", "isotonik_periode", FEATURE_COLUMNS),
            ("neural_net_mlp+oof", "mlp", "isotonik_oof", FEATURE_COLUMNS)]
    probs = {}
    for name, kind, calib, cols in runs:
        t0 = time.time()
        probs[name] = fit_predict(kind, calib, cols, ds)
        rows[name] = score(probs[name], te, ref)
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    probs["ensemble(gbm+mlp+logit)+oof"] = np.mean([probs["gbm+oof"], probs["neural_net_mlp+oof"],
                                                     probs["logistik+oof"]], axis=0)
    rows["ensemble(gbm+mlp+logit)+oof"] = score(probs["ensemble(gbm+mlp+logit)+oof"], te, ref)
    print_table(rows, "A. Perbandingan model (M>=5, 7 hari, sel 1°, uji 1995-2007)")
    save("A_models", {k: v for k, v in rows.items()})
    np.savez_compressed(os.path.join(OUT, "A_probs.npz"), y=te["y"].to_numpy(), ref=ref, **probs)
    plot_reliability_multi(rows, ["klimatologi", "gbm", "gbm+oof", "neural_net_mlp+oof",
                                  "ensemble(gbm+mlp+logit)+oof"], os.path.join(OUT, "A_reliability.png"))


def exp_ablation():
    ds = dataset(FeatureConfig(**BASE_CFG))
    te, ref = ds["test"], climatology(ds)
    rows = {}
    for name, cols in FEATURE_SETS.items():
        rows[name] = score(fit_predict("gbm", "isotonik_oof", cols, ds), te, ref)
        print(f"  {name} selesai", flush=True)
    print_table(rows, "B. Ablasi fitur (GBM + kalibrasi OOF)")
    save("B_ablation", rows)


def _target_job(args):
    m, w, c = args
    cfg = FeatureConfig(target_mag=m, window_days=w, cell_deg=c, step_days=max(1.0, w / 3.5))
    ds = dataset(cfg)
    te, ref = ds["test"], climatology(ds)
    return f"M>={m} {w:g}h sel{c:g}°", score(fit_predict("gbm", "isotonik_oof", FEATURE_COLUMNS, ds), te, ref)


def exp_targets():
    jobs = [(m, w, c) for m in (5.0, 6.0) for w in (1.0, 7.0, 30.0) for c in (1.0, 2.0)]
    rows = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for name, r in ex.map(_target_job, jobs):
            rows[name] = r
            print(f"  {name} selesai", flush=True)
    print_table(rows, "C. Definisi target (GBM + kalibrasi OOF, uji 1995-2007)")
    save("C_targets", rows)


def _rolling_job(split):
    ds = dataset(FeatureConfig(**BASE_CFG), split=split)
    te = ds["test"]
    end = split.get("test_end")
    ref = climatology(ds)
    name = f"uji {split['cal_end'][:4]}-{(end or '2008')[:4]}"
    return name, score(fit_predict("gbm", "isotonik_oof", FEATURE_COLUMNS, ds), te, ref)


def exp_rolling():
    splits = [dict(train_end="1955-01-01", cal_end="1965-01-01", test_end="1975-01-01"),
              dict(train_end="1965-01-01", cal_end="1975-01-01", test_end="1985-01-01"),
              dict(train_end="1975-01-01", cal_end="1985-01-01", test_end="1995-01-01"),
              dict(train_end="1985-01-01", cal_end="1995-01-01")]
    rows = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for name, r in ex.map(_rolling_job, splits):
            rows[name] = r
            print(f"  {name} selesai", flush=True)
    print_table(rows, "D. Uji bergulir (GBM + kalibrasi OOF; latih = semua data sebelum periode uji)")
    save("D_rolling", rows)


def plot_reliability_multi(rows, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="kalibrasi sempurna")
    for n in names:
        rel = [r for r in rows[n]["reliability"] if r["n"] >= 5]
        ax.plot([r["mean_pred"] for r in rel], [r["observed"] for r in rel], marker="o", label=n)
    ax.axvspan(0.8, 1, color="#2da44e", alpha=0.08)
    ax.set_xlabel("Peluang yang diprakirakan"); ax.set_ylabel("Frekuensi kejadian sebenarnya")
    ax.set_title("Reliabilitas periode uji 1995-2007 (JMA)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    {"models": exp_models, "ablation": exp_ablation, "targets": exp_targets,
     "rolling": exp_rolling}[sys.argv[1]]()
