"""Model probabilitas + kalibrasi isotonik, dan baseline klimatologi."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class CalibratedModel:
    """Klasifier (GBM / logistik) yang probabilitasnya dikalibrasi ulang pada periode terpisah.

    Kalibrasi inilah yang membuat angka "80%" bermakna: dari semua prakiraan ~80%,
    sekitar 80% di antaranya memang terjadi (diuji ulang di periode uji).
    """

    def __init__(self, kind: str = "gbm", random_state: int = 0):
        self.kind = kind
        if kind == "gbm":
            self.base = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                random_state=random_state)
        elif kind == "logit":
            self.base = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        elif kind == "mlp":
            self.base = make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512, learning_rate_init=1e-3,
                max_iter=60, early_stopping=True, validation_fraction=0.15, n_iter_no_change=6,
                random_state=random_state))
        else:
            raise ValueError(kind)
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.base.fit(X_train, y_train)
        self.iso.fit(self.base.predict_proba(X_cal)[:, 1], y_cal)
        return self

    def predict_raw(self, X):
        return self.base.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.iso.predict(self.predict_raw(X))


class OOFCalibratedModel(CalibratedModel):
    """Kalibrasi dari prediksi out-of-fold berblok waktu atas SELURUH periode latih+kalibrasi.

    Periode kalibrasi tunggal (mis. 10 tahun) hanya berisi sedikit sekuens gempa besar, sehingga
    bin peluang tinggi dikalibrasi dari segelintir kasus. Dengan OOF, setiap sekuens historis
    ikut menyumbang contoh kalibrasi, lalu model akhir dilatih pada semua data sebelum periode uji.
    """

    def __init__(self, kind="gbm", n_folds=5, random_state=0):
        super().__init__(kind, random_state)
        self.n_folds = n_folds

    def fit_oof(self, X, y, t):
        from sklearn.base import clone
        X, y, t = X.reset_index(drop=True), np.asarray(y), np.asarray(t)
        edges = np.quantile(t, np.linspace(0, 1, self.n_folds + 1))
        fold = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, self.n_folds - 1)
        oof = np.zeros(len(y))
        for k in range(self.n_folds):
            m = fold == k
            est = clone(self.base).fit(X[~m], y[~m])
            oof[m] = est.predict_proba(X[m])[:, 1]
        self.iso.fit(oof, y)
        self.base.fit(X, y)
        return self


class Climatology:
    """Baseline: peluang = frekuensi historis per sel (tanpa informasi waktu luruh)."""

    def fit(self, cells, y, n_cells, prior_weight=5.0):
        cells = np.asarray(cells)
        y = np.asarray(y)
        glob = y.mean()
        hits = np.bincount(cells, weights=y, minlength=n_cells)
        n = np.bincount(cells, minlength=n_cells)
        self.rate = (hits + prior_weight * glob) / (n + prior_weight)
        return self

    def predict_proba(self, cells):
        return self.rate[np.asarray(cells)]


def feature_importance_proxy(model: CalibratedModel, X: pd.DataFrame, y, n_repeats=3, seed=0):
    """Permutation importance sederhana berbasis Brier score."""
    rng = np.random.default_rng(seed)
    base = np.mean((model.predict_proba(X) - y) ** 2)
    out = {}
    for col in X.columns:
        s = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            s.append(np.mean((model.predict_proba(Xp) - y) ** 2) - base)
        out[col] = float(np.mean(s))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
