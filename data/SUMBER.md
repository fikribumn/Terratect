# Sumber data

`jma_1926_2007_M4.5.csv` — katalog gempa dangkal (kedalaman < 100 km), M >= 4.5 (magnitudo JMA),
1926-01-08 s/d 2007-12-29, wilayah 128-145°BT, 27-45°LU.

- Data asli: Japan Meteorological Agency (JMA), dihimpun oleh Jiancang Zhuang (ISM Tokyo)
  untuk perangkat lunak ETAS: https://bemlar.ism.ac.jp/zhuang/software.html
- Diambil dari dataset `japan.quakes` pada paket R `ETAS` (CRAN, lisensi GPL >= 2),
  lalu dikonversi ke CSV (kolom time UTC, latitude, longitude, depth km positif, mag).
- Referensi: Zhuang, Ogata & Vere-Jones (2002), JASA 97(458), 369-380.
