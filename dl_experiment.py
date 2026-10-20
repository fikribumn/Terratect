#!/usr/bin/env python3
"""Eksperimen E: deep learning sekuens (PyTorch) vs GBM pada data uji yang sama.

Input model = urutan 32 gempa terakhir (<= 3 tahun) dalam radius 200 km dari pusat sel:
  [log waktu lalu, M - Mc, jarak, kedalaman, dlat, dlon, di-sel-sama]
ditambah fitur statis. Tidak ada fitur waktu luruh buatan tangan pada varian "murni";
model harus menemukan sendiri pola peluruhan dari urutan event mentah.

  python dl_experiment.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments import (BASE_CFG, CATALOG, GEOLOGY, MC, OUT, climatology, dataset, fit_predict,
                         print_table, score)
from gempa.data import load_catalog
from gempa.features import FEATURE_COLUMNS, FeatureConfig
from gempa.geology import haversine
from sklearn.isotonic import IsotonicRegression

SEQ_LEN, RADIUS_KM, MAX_AGE = 32, 200.0, 3 * 365.25
DL_NEG_FRAC = 0.05  # CPU saja: latih dengan 5% sampel negatif (bobot 20x)
torch.set_num_threads(4)


def build_sequences(rows: pd.DataFrame, cat: pd.DataFrame, grid):
    t, mag = cat["t"].to_numpy(), cat["mag"].to_numpy()
    lat, lon, dep = cat["latitude"].to_numpy(), cat["longitude"].to_numpy(), cat["depth"].to_numpy(float)
    cell_of_ev = grid.cell_of(lat, lon)
    dist = haversine(grid.lat[:, None], grid.lon[:, None], lat[None, :], lon[None, :])
    near = [np.where(dist[c] <= RADIUS_KM)[0] for c in range(len(grid))]  # terurut waktu
    X = np.zeros((len(rows), SEQ_LEN, 7), np.float32)
    M = np.zeros((len(rows), SEQ_LEN), bool)  # True = padding
    M[:] = True
    for r, (it, c) in enumerate(zip(rows["issue_t"].to_numpy(), rows["cell"].to_numpy())):
        idx = near[c]
        k = np.searchsorted(t[idx], it)
        sel = idx[max(0, k - SEQ_LEN):k]
        sel = sel[it - t[sel] <= MAX_AGE]
        n = len(sel)
        if n == 0:
            continue
        X[r, -n:, 0] = np.log10(it - t[sel] + 1e-3) / 3.0
        X[r, -n:, 1] = mag[sel] - MC
        X[r, -n:, 2] = dist[c, sel] / RADIUS_KM
        X[r, -n:, 3] = np.nan_to_num(dep[sel], nan=30.0) / 100.0
        X[r, -n:, 4] = (lat[sel] - grid.lat[c]) / 2.0
        X[r, -n:, 5] = (lon[sel] - grid.lon[c]) / 2.0
        X[r, -n:, 6] = cell_of_ev[sel] == c
        M[r, -n:] = False
    return X, M


class SeqModel(nn.Module):
    def __init__(self, n_static, kind="gru", d=32):
        super().__init__()
        self.kind = kind
        self.inp = nn.Sequential(nn.Linear(7, d), nn.GELU())
        if kind == "gru":
            self.enc = nn.GRU(d, d, batch_first=True)
        else:
            self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
        self.static = nn.Sequential(nn.Linear(n_static, d), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, x, pad, s):
        h = self.inp(x)
        empty = pad.all(1)
        if self.kind == "gru":
            out, _ = self.enc(h)
            last = out[:, -1]
        else:
            pad2 = pad.clone()
            pad2[empty, -1] = False  # hindari NaN untuk sekuens kosong
            out = self.enc(h + self.pos, src_key_padding_mask=pad2)
            last = out[:, -1]
        w = (~pad).float().unsqueeze(-1)
        mean = (out * w).sum(1) / w.sum(1).clamp(min=1)
        last = torch.where(empty.unsqueeze(-1), torch.zeros_like(last), last)
        return self.head(torch.cat([last, mean, self.static(s)], -1)).squeeze(-1)


def train_model(kind, Xs, Ms, S, y, w, val_frac=0.15, epochs=20, patience=3, seed=0):
    torch.manual_seed(seed)
    n = len(y)
    n_val = int(n * val_frac)  # validasi = bagian paling akhir (baris terurut waktu)
    tr, va = slice(0, n - n_val), slice(n - n_val, n)
    model = SeqModel(S.shape[1], kind)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    Xt, Mt, St, yt, wt = T(Xs), T(Ms), T(S.astype(np.float32)), T(y.astype(np.float32)), T(w.astype(np.float32))
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n - n_val)
        t0 = time.time()
        for b in range(0, len(perm), 1024):
            i = perm[b:b + 1024]
            loss = (lossf(model(Xt[i], Mt[i], St[i]), yt[i]) * wt[i]).sum() / wt[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = (lossf(model(Xt[va], Mt[va], St[va]), yt[va]) * wt[va]).sum().item() / wt[va].sum().item()
        print(f"    [{kind}] epoch {ep + 1}: val loss {lv:.5f} ({time.time() - t0:.0f}s)", flush=True)
        if lv < best - 1e-5:
            best, bad = lv, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, X, M, S):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 4096):
            sl = slice(b, b + 4096)
            out.append(torch.sigmoid(model(torch.from_numpy(X[sl]), torch.from_numpy(M[sl]),
                                           torch.from_numpy(S[sl].astype(np.float32)))).numpy())
    return np.concatenate(out)


def main():
    ds = dataset(FeatureConfig(**BASE_CFG))
    cat = load_catalog(CATALOG)
    cat = cat[cat["mag"] >= MC].reset_index(drop=True)
    tr, ca, te = ds["train"], ds["cal"], ds["test"]
    rng = np.random.default_rng(0)
    tr = tr[(tr["y"] == 1).to_numpy() | (rng.random(len(tr)) < DL_NEG_FRAC)].sort_values("issue_t")
    w_tr = np.where(tr["y"] == 1, 1.0, 1.0 / DL_NEG_FRAC)
    t0 = time.time()
    seq = {name: build_sequences(d, cat, ds["grid"]) for name, d in (("tr", tr), ("ca", ca), ("te", te))}
    print(f"Sekuens dibangun: {sum(len(v[0]) for v in seq.values()):,} sampel ({time.time() - t0:.0f}s)", flush=True)

    ref = climatology(ds)
    rows = {"klimatologi": score(ref, te, ref),
            "gbm (kalibrasi periode)": score(fit_predict("gbm", "isotonik_periode", FEATURE_COLUMNS, ds), te, ref)}
    variants = {"murni": GEOLOGY, "hibrida": FEATURE_COLUMNS}
    probs = {}
    for vname, cols in variants.items():
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        S = {k: ((d[cols] - mu) / sd).to_numpy() for k, d in (("tr", tr), ("ca", ca), ("te", te))}
        for kind in ("gru", "transformer"):
            name = f"{kind}_{vname}"
            t0 = time.time()
            model = train_model(kind, *seq["tr"], S["tr"], tr["y"].to_numpy(), w_tr)
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(predict(model, *seq["ca"], S["ca"]), ca["y"])
            probs[name] = iso.predict(predict(model, *seq["te"], S["te"]))
            rows[name] = score(probs[name], te, ref)
            print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    gbm = fit_predict("gbm", "isotonik_oof", FEATURE_COLUMNS, ds)
    rows["gbm+oof"] = score(gbm, te, ref)
    rows["ensemble gbm+oof & transformer_hibrida"] = score((gbm + probs["transformer_hibrida"]) / 2, te, ref)
    print_table(rows, "E. Deep learning sekuens vs GBM (M>=5, 7 hari, sel 1°, uji 1995-2007)")
    with open(os.path.join(OUT, "E_deep_learning.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)


if __name__ == "__main__":
    main()
