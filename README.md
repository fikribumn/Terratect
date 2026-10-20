# Terratect — Prakiraan Probabilistik Gempa Wilayah Jepang

> **Hasil eksperimen pada data JMA 1926–2007 (GBM, neural network, GRU, Transformer, dll.):
> lihat [LAPORAN_EKSPERIMEN.md](LAPORAN_EKSPERIMEN.md).**

Tools ini memperkirakan **peluang** terjadinya gempa M ≥ X di tiap sel grid wilayah Jepang dalam
N hari ke depan. Peluang itu lalu **diuji ulang (backtest)** di periode yang tidak pernah dilihat
model, sehingga angka seperti "80%" bisa diperiksa: dari semua prakiraan ≥ 80%, berapa persen yang
benar-benar terjadi?

> **Catatan jujur.** Memprediksi *gempa utama* (kapan & di mana tepatnya) dengan keyakinan 80% belum
> bisa dilakukan oleh metode ilmiah mana pun. Yang bisa dicapai: peluang tinggi (≥ 80%) muncul
> terutama **setelah gempa besar**, saat aktivitas susulan masih tinggi dan meluruh mengikuti
> hukum Omori-Utsu. Di luar kondisi itu peluang biasanya hanya beberapa persen. Tools ini
> melaporkan hal tersebut apa adanya lewat diagram reliabilitas dan status `VALID / BELUM CUKUP
> BUKTI / TIDAK VALID`.

## Data

Data yang kamu maksud kemungkinan **katalog JMA** (Japan Meteorological Agency, *Unified Hypocenter
Catalog*) atau jaringan **NIED Hi-net / K-NET**. Keduanya butuh unduh manual / registrasi.
Default tools ini memakai **USGS ComCat** (FDSN, gratis, tanpa login) yang dibatasi kotak wilayah
Jepang (lintang 24–46°, bujur 122–148°).

Katalog JMA/NIED juga bisa dipakai: konversi ke CSV dengan kolom
`time, latitude, longitude, depth, mag`.

Repo ini sudah menyertakan **katalog JMA 1926–2007 (M ≥ 4,5, kedalaman < 100 km)** di
`data/jma_1926_2007_M4.5.csv` (sumber: [data/SUMBER.md](data/SUMBER.md)), yang dipakai untuk eksperimen.

## Metode

| Faktor | Isi |
|---|---|
| **Waktu luruh** | Parameter Omori-Utsu `n(t) = K/(t+c)^p` di-fit (MLE, Ogata 1983) dari sekuens susulan M ≥ 6 di data latih. Fitur: ekspektasi jumlah susulan di jendela prakiraan, dijumlahkan dari semua gempa sebelumnya dengan produktivitas `10^(α(M−Mc))` dan kernel spasial selebar panjang rupture. Ada varian p cepat/lambat. |
| **Aktivitas terbaru** | Jumlah event dalam 100 km pada 1/7/30/365 hari terakhir, M maksimum 30 hari, waktu sejak gempa ≥ target terakhir. |
| **Keadaan geologi** | Laju latar (setelah dekluster Gardner-Knopoff), b-value lokal (Aki-Utsu), rata-rata kedalaman & fraksi gempa dalam (>70 km), jarak ke palung (Kuril, Jepang, Izu-Bonin, Sagami, Nankai, Ryukyu) dan sesar/zona daratan (MTL, ISTL, tepi Laut Jepang). Semua dihitung **hanya dari periode latih**. |
| **Model** | Gradient boosting (atau regresi logistik) → **kalibrasi isotonik** di periode terpisah. |
| **Validasi** | Periode uji terpisah: AUC, Brier, Brier Skill Score vs klimatologi, diagram reliabilitas, presisi & recall saat peluang ≥ 80% dengan interval kepercayaan Wilson 95%. |

Pembagian waktu default: latih s/d 2016, kalibrasi 2016–2019, uji 2020–sekarang (termasuk gempa
Noto 2024).

## Cara pakai

```bash
pip install -r requirements.txt

# 1. Unduh katalog (USGS, M>=4 sejak 2000)
python gempa_tool.py fetch --start 2000-01-01 --minmag 4.0 --out data/japan_usgs.csv

# 2. Backtest: latih, kalibrasi, uji -> hasil/report.json, reliability.png, model.pkl
python gempa_tool.py backtest --catalog data/japan_usgs.csv --out hasil \
    --target-mag 5.0 --window-days 7 --step-days 1 --cell-deg 1.0

# 3. Prakiraan terkini (jendela 7 hari setelah event terakhir di katalog)
python gempa_tool.py fetch --start 2000-01-01 --out data/japan_usgs.csv   # perbarui data
python gempa_tool.py predict --model hasil/model.pkl --catalog data/japan_usgs.csv --out hasil
```

Opsi penting:
- `--target-mag`, `--window-days`, `--cell-deg`: definisi kejadian yang diprakirakan. Sel lebih besar
  / jendela lebih panjang / magnitudo lebih kecil → base rate naik → lebih banyak alarm ≥ 80%.
- `--step-days 1`: lebih banyak sampel uji (jendela saling tumpang tindih, jadi sampel tidak
  sepenuhnya independen; CI sedikit terlalu sempit).
- `--model logit` / `--model mlp`: regresi logistik / neural network sebagai pembanding.
- `--mc`: magnitudo kelengkapan manual (default otomatis, maximum curvature + 0.2).

Contoh dengan data JMA yang disertakan:

```bash
python gempa_tool.py backtest --catalog data/jma_1926_2007_M4.5.csv --mc 4.5 \
    --train-end 1985-01-01 --cal-end 1995-01-01 --step-days 2 --out hasil/jma
```

Eksperimen lengkap: `experiments.py` (model, ablasi, target, uji bergulir) dan
`dl_experiment.py` (GRU/Transformer, butuh `torch`).

`python gempa_tool.py synthetic` membuat katalog ETAS **sintetis** hanya untuk menguji bahwa kode
berjalan — hasilnya tidak bermakna untuk Jepang.

## Membaca hasil

- **Diagram reliabilitas**: titik harus dekat garis diagonal. Titik di zona hijau (≥ 80%) yang
  berada ≥ 0.8 berarti klaim "80%" didukung data.
- **Recall** saat ≥ 80% hampir pasti kecil: sebagian besar gempa M5+ tetap terjadi tanpa peringatan
  peluang tinggi. Itu batas fisis, bukan bug.
- Status `BELUM CUKUP BUKTI` muncul bila alarm ≥ 80% kurang dari 30 kasus.

Tools ini untuk riset/edukasi. Untuk keselamatan, ikuti informasi resmi JMA (緊急地震速報 dan
informasi gempa susulan).
