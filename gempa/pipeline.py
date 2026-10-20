"""Alur backtest (latih -> kalibrasi -> uji) dan prakiraan terkini."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from .data import from_days, load_catalog, to_days
from .evaluate import evaluate, format_report, verdict
from .features import FEATURE_COLUMNS, DynamicBuilder, FeatureConfig, Grid, assemble, static_features
from .geology import TECTONIC_LINES, b_value_mle, mc_maxc
from .model import CalibratedModel, Climatology, feature_importance_proxy
from .omori import fit_sequences, pooled_parameters


def _split_times(t0, t1, train_end, cal_end, warmup_days=365.0):
    start = t0 + warmup_days
    if train_end is None or cal_end is None:
        span = t1 - start
        return start, start + 0.6 * span, start + 0.8 * span
    return start, to_days(train_end), to_days(cal_end)


def build_dataset(catalog, cfg: FeatureConfig, train_end=None, cal_end=None, mc=None, main_mag=6.0,
                  start_date=None, end_date=None, verbose=True):
    """Muat katalog, fit Omori, hitung fitur, dan bagi latih/kalibrasi/uji berdasarkan waktu."""
    log = print if verbose else (lambda *a, **k: None)
    cat = load_catalog(catalog) if isinstance(catalog, str) else catalog
    if start_date:
        cat = cat[cat["t"] >= to_days(start_date)]
    if end_date:
        cat = cat[cat["t"] < to_days(end_date)]
    cat = cat.reset_index(drop=True)
    t0, t1 = cat["t"].min(), cat["t"].max()
    start, tr_end, ca_end = _split_times(t0, t1, train_end, cal_end)
    train_cat = cat[cat["t"] < tr_end]

    mc = mc if mc is not None else mc_maxc(train_cat["mag"])
    if cfg.target_mag < mc:
        raise ValueError(f"target_mag {cfg.target_mag} < Mc {mc}: katalog tidak lengkap untuk target ini")
    cat = cat[cat["mag"] >= mc].reset_index(drop=True)
    train_cat = cat[cat["t"] < tr_end]
    global_b, sb = b_value_mle(train_cat["mag"], mc)
    log(f"[1] Katalog: {len(cat):,} event M>={mc:.1f} | {from_days(t0).date()} s/d {from_days(t1).date()}")
    log(f"    b-value regional = {global_b:.2f} ± {sb:.2f}")
    log(f"    Latih: s/d {from_days(tr_end).date()} | Kalibrasi: s/d {from_days(ca_end).date()} | Uji: s/d {from_days(t1).date()}")

    fits = fit_sequences(train_cat, mc, main_mag=main_mag)
    om = pooled_parameters(fits)
    log(f"[2] Peluruhan Omori-Utsu dari {om['n_sequences']} sekuens susulan: p={om['p']:.2f}, c={om['c']:.3f} hari")

    grid = Grid(cfg, train_cat)
    static = static_features(grid, train_cat, mc, global_b)
    log(f"[3] Grid {cfg.cell_deg}° : {len(grid)} sel aktif; fitur geologi dihitung dari periode latih")

    dyn_builder = DynamicBuilder(cat, grid, cfg, mc, om["c"], om["p"])
    issue = np.arange(start, t1 - cfg.window_days, cfg.step_days)
    df = assemble(dyn_builder.build(issue), static)
    log(f"[4] {len(df):,} baris fitur ({len(issue)} waktu terbit x {len(grid)} sel)")

    # label periode latih tidak boleh melewati batas -> buang baris yang jendelanya menyeberang
    tr = df[df["issue_t"] + cfg.window_days <= tr_end]
    ca = df[(df["issue_t"] >= tr_end) & (df["issue_t"] + cfg.window_days <= ca_end)]
    te = df[df["issue_t"] >= ca_end]
    return dict(cat=cat, cfg=cfg, mc=mc, b_value=global_b, omori=om, fits=fits, grid=grid, static=static,
                train=tr, cal=ca, test=te, start=start, train_end=tr_end, cal_end=ca_end, end=t1)


def run_backtest(catalog_path, out_dir, cfg: FeatureConfig, train_end=None, cal_end=None,
                 mc=None, kind="gbm", threshold=0.8, main_mag=6.0):
    os.makedirs(out_dir, exist_ok=True)
    ds = build_dataset(catalog_path, cfg, train_end, cal_end, mc=mc, main_mag=main_mag)
    tr, ca, te, grid, static = ds["train"], ds["cal"], ds["test"], ds["grid"], ds["static"]
    mc, global_b, om, fits = ds["mc"], ds["b_value"], ds["omori"], ds["fits"]
    start, tr_end, ca_end, t1 = ds["start"], ds["train_end"], ds["cal_end"], ds["end"]

    model = CalibratedModel(kind).fit(tr[FEATURE_COLUMNS], tr["y"], ca[FEATURE_COLUMNS], ca["y"])
    clim = Climatology().fit(pd.concat([tr, ca])["cell"], pd.concat([tr, ca])["y"], len(grid))

    p_te = model.predict_proba(te[FEATURE_COLUMNS])
    p_clim = clim.predict_proba(te["cell"])
    res = evaluate(p_te, te["y"].to_numpy(), p_ref=p_clim, threshold=threshold)
    res_clim = evaluate(p_clim, te["y"].to_numpy(), threshold=threshold)
    res["verdict"] = verdict(res, target=threshold)
    imp = feature_importance_proxy(model, te[FEATURE_COLUMNS].sample(min(len(te), 60000), random_state=0),
                                   te["y"].sample(min(len(te), 60000), random_state=0).to_numpy())

    print()
    print(format_report(f"MODEL ({kind}) - periode uji", res))
    print(format_report("BASELINE klimatologi - periode uji", res_clim))
    print("\n  Kontribusi fitur (kenaikan Brier saat diacak):")
    for k, v in list(imp.items())[:8]:
        print(f"    {k:<24} {v:+.6f}")
    print(f"\n>>> KESIMPULAN: {res['verdict']}")

    te_out = te[["issue_t", "cell", "cell_lat", "cell_lon", "y"]].copy()
    te_out["issue_time"] = te_out["issue_t"].map(lambda d: from_days(d).isoformat())
    te_out["p_model"], te_out["p_climatology"] = p_te, p_clim
    te_out.drop(columns="issue_t").to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    report = dict(config=cfg.to_dict(), mc=mc, b_value=global_b, omori=om, omori_sequences=fits,
                  splits=dict(start=str(from_days(start)), train_end=str(from_days(tr_end)),
                              cal_end=str(from_days(ca_end)), end=str(from_days(t1))),
                  model=res, climatology=res_clim, feature_importance=imp)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    plot_reliability(res, res_clim, os.path.join(out_dir, "reliability.png"))

    bundle = dict(cfg=cfg, mc=mc, omori=om, grid=grid, static=static, model=model, kind=kind,
                  reliability=res["reliability"], verdict=res["verdict"], threshold=threshold)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nTersimpan di {out_dir}/: model.pkl, report.json, reliability.png, test_predictions.csv")
    return report


def run_predict(model_path, catalog_path, out_dir, at=None, top=15):
    with open(model_path, "rb") as f:
        b = pickle.load(f)
    cfg, grid = b["cfg"], b["grid"]
    cat = load_catalog(catalog_path)
    cat = cat[cat["mag"] >= b["mc"]].reset_index(drop=True)
    it = to_days(at) if at else cat["t"].max() + 1e-3
    dyn = DynamicBuilder(cat, grid, cfg, b["mc"], b["omori"]["c"], b["omori"]["p"]).build([it], with_labels=False)
    df = assemble(dyn, b["static"])
    df["probability"] = b["model"].predict_proba(df[FEATURE_COLUMNS])
    df = df.sort_values("probability", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["cell_lat", "cell_lon", "probability", "log_omori", "n_7d", "max_mag_30d", "b_value", "dist_trench_km"]
    df[cols].to_csv(os.path.join(out_dir, "forecast.csv"), index=False)

    t_from = from_days(it)
    t_to = from_days(it + cfg.window_days)
    print(f"Prakiraan peluang gempa M>={cfg.target_mag} per sel {cfg.cell_deg}°")
    print(f"Jendela: {t_from:%Y-%m-%d %H:%M} s/d {t_to:%Y-%m-%d %H:%M} UTC\n")
    print(f"{'lat':>6} {'lon':>7} {'peluang':>8}  {'kejadian 7h':>11} {'Mmaks 30h':>9}  keandalan historis bin ini")
    for _, r in df.head(top).iterrows():
        rel = _reliability_of(b["reliability"], r["probability"])
        print(f"{r.cell_lat:6.1f} {r.cell_lon:7.1f} {r.probability:8.1%}  {int(r.n_7d):>11} {r.max_mag_30d:9.1f}  {rel}")
    n_alarm = int((df["probability"] >= b["threshold"]).sum())
    print(f"\nSel dengan peluang >= {b['threshold']:.0%}: {n_alarm}")
    print(f"Status validasi backtest model ini: {b['verdict']}")
    plot_map(df, cat, it, cfg, os.path.join(out_dir, "forecast_map.png"))
    print(f"Tersimpan: {out_dir}/forecast.csv, {out_dir}/forecast_map.png")
    return df


def _reliability_of(table, p):
    for r in table:
        lo, hi = map(float, r["bin"].split("-"))
        if lo <= p < hi + 1e-9 and r["n"]:
            return f"terjadi {r['observed']:.0%} dari {r['n']:,} kasus serupa"
    return "-"


def plot_reliability(res, res_clim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="kalibrasi sempurna")
    for r_, label, color in ((res, "model", "#1f6feb"), (res_clim, "klimatologi", "#d29922")):
        rows = [r for r in r_["reliability"] if r["n"]]
        x = [r["mean_pred"] for r in rows]
        y = [r["observed"] for r in rows]
        err = np.array([[r["observed"] - r["ci95"][0], r["ci95"][1] - r["observed"]] for r in rows]).T
        ax.errorbar(x, y, yerr=err, marker="o", color=color, label=label, capsize=3)
    ax.axvspan(res["threshold"], 1, color="#2da44e", alpha=0.08, label=f"zona alarm >= {res['threshold']:.0%}")
    ax.set_xlabel("Peluang yang diprakirakan")
    ax.set_ylabel("Frekuensi kejadian sebenarnya")
    ax.set_title("Diagram reliabilitas (periode uji)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_map(df, cat, it, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 8))
    recent = cat[(cat["t"] < it) & (cat["t"] >= it - 30)]
    ax.scatter(cat["longitude"], cat["latitude"], s=0.3, color="0.85", label="semua event")
    sc = ax.scatter(df["cell_lon"], df["cell_lat"], c=df["probability"], s=(cfg.cell_deg * 22) ** 2,
                    marker="s", cmap="inferno_r", vmin=0, vmax=1, alpha=0.75)
    ax.scatter(recent["longitude"], recent["latitude"], s=4 * 2 ** (recent["mag"] - 3), facecolor="none",
               edgecolor="#1f6feb", lw=0.8, label="event 30 hari terakhir")
    for name, line in TECTONIC_LINES.items():
        xs, ys = zip(*line)
        ax.plot(xs, ys, color="#cf222e" if "trench" in name or "trough" in name else "#8250df", lw=1.2)
    plt.colorbar(sc, ax=ax, label=f"Peluang M>={cfg.target_mag} dalam {cfg.window_days:g} hari")
    ax.set_xlabel("Bujur"); ax.set_ylabel("Lintang")
    ax.set_title(f"Prakiraan per {from_days(it):%Y-%m-%d}")
    ax.set_aspect(1.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
