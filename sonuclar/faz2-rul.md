# Faz 2 — RUL (kalan faydalı ömür) regresyonu

- kaynak: `sentetik-disk`  ·  pencere: 30 çevrim  ·  train birim: 43, val: 7, test: 10
- değerlendirme: resmi protokol = birim başına yalnız son pencere (literatürle karşılaştırılabilir); kaydırmalı = eğitim-tarzı, bilgi amaçlı

## Resmi protokol (birim başına 1 tahmin)

| model | MAE (çevrim) | RMSE | C-MAPSS skoru (düşük iyi) |
|---|---:|---:|---:|
| rul_gru | 9.93 | 14.24 | 39.0 |
| ort_taban (train ortalaması) | 76.06 | 76.06 | 20099.1 |

**Baseline'ı geçti: EVET** (MAE karşılaştırması)

## Kaydırmalı pencere (bilgi amaçlı, çok örnek)

n=1825 · MAE 13.70 · RMSE 17.18 · C-MAPSS skoru 9636.7 (örtüşen pencereler nedeniyle skor resmi protokolden daha büyük görünür — beklenen)

NOT: sentetik veride sonuçlar boru hattı doğrulaması içindir; gerçek C-MAPSS (python 00_veri_indir.py --cmapss-bilgi) ile literatürle karşılaştırılabilir.