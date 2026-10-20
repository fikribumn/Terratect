"""Metrik validasi prakiraan probabilistik."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

BINS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z ** 2 / n
    mid = (ph + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(mid - half), float(mid + half))


def reliability_table(p, y, bins=BINS):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        n, k = int(m.sum()), int(y[m].sum())
        rows.append(dict(bin=f"{lo:.2f}-{min(hi, 1):.2f}", n=n, mean_pred=float(p[m].mean()) if n else np.nan,
                         observed=k / n if n else np.nan, ci95=wilson_ci(k, n)))
    return rows


def evaluate(p, y, p_ref=None, threshold=0.8):
    p, y = np.asarray(p, float), np.asarray(y, int)
    brier = float(np.mean((p - y) ** 2))
    res = dict(n=int(len(y)), positives=int(y.sum()), base_rate=float(y.mean()), brier=brier,
               auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
               log_loss=float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))))
    if p_ref is not None:
        bref = float(np.mean((np.asarray(p_ref) - y) ** 2))
        res["brier_ref"] = bref
        res["brier_skill_score"] = 1 - brier / bref if bref > 0 else np.nan
    alarm = p >= threshold
    k, n = int(y[alarm].sum()), int(alarm.sum())
    res["threshold"] = threshold
    res["alarms"] = n
    res["alarm_hits"] = k
    res["precision_at_threshold"] = k / n if n else np.nan
    res["precision_ci95"] = wilson_ci(k, n)
    res["recall_at_threshold"] = k / int(y.sum()) if y.sum() else np.nan
    res["reliability"] = reliability_table(p, y)
    return res


def verdict(res, target=0.8, min_alarms=30):
    """Apakah klaim 'prakiraan >= target benar terjadi >= target' didukung data uji?"""
    n, prec, (lo, _) = res["alarms"], res["precision_at_threshold"], res["precision_ci95"]
    if n == 0:
        return "TIDAK TERUJI: model tidak pernah mengeluarkan peluang >= {:.0%} di periode uji.".format(target)
    if n < min_alarms:
        return (f"BELUM CUKUP BUKTI: hanya {n} alarm >= {target:.0%} (presisi {prec:.0%}, "
                f"CI95 {res['precision_ci95'][0]:.0%}-{res['precision_ci95'][1]:.0%}). Butuh >= {min_alarms}.")
    if lo >= target - 0.1 and prec >= target:
        return f"VALID: {n} alarm >= {target:.0%}, {prec:.0%} benar terjadi (CI95 bawah {lo:.0%})."
    if prec >= target:
        return f"CENDERUNG VALID: presisi {prec:.0%} dari {n} alarm, tetapi CI95 bawah {lo:.0%} masih lebar."
    return f"TIDAK VALID: dari {n} alarm >= {target:.0%}, hanya {prec:.0%} yang terjadi."


def format_report(name, res):
    lines = [f"== {name} ==",
             f"  sampel={res['n']:,}  positif={res['positives']:,}  base rate={res['base_rate']:.4f}",
             f"  AUC={res['auc']:.3f}  Brier={res['brier']:.5f}  LogLoss={res['log_loss']:.4f}"]
    if "brier_skill_score" in res:
        lines.append(f"  Brier Skill Score vs klimatologi = {res['brier_skill_score']:+.3f}")
    lines.append(f"  Alarm P>={res['threshold']:.0%}: {res['alarms']} | terjadi {res['alarm_hits']} "
                 f"| presisi {res['precision_at_threshold']:.1%} | recall {res['recall_at_threshold']:.1%}")
    lines.append("  Reliabilitas (prediksi vs kenyataan):")
    for r in res["reliability"]:
        if r["n"]:
            lines.append(f"    {r['bin']:>10}  n={r['n']:>7,}  prediksi={r['mean_pred']:.3f}  "
                         f"terjadi={r['observed']:.3f}  CI95=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]")
    return "\n".join(lines)
