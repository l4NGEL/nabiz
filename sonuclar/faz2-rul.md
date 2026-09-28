# Faz 2 — RUL (kalan faydalı ömür) regresyonu

- kaynak: `gercek-FD001`  ·  pencere: 30 çevrim  ·  train birim: 88, val: 12, test: 100
- değerlendirme: resmi protokol = birim başına yalnız son pencere (literatürle karşılaştırılabilir); kaydırmalı = eğitim-tarzı, bilgi amaçlı

## Resmi protokol (birim başına 1 tahmin)

| model | MAE (çevrim) | RMSE | C-MAPSS skoru (düşük iyi) |
|---|---:|---:|---:|
| rul_gru | 11.48 | 15.00 | 538.2 |
| ort_taban (train ortalaması) | 35.19 | 40.40 | 16673.3 |

**Baseline'ı geçti: EVET** (MAE karşılaştırması)

## Kaydırmalı pencere (bilgi amaçlı, çok örnek)

n=10196 · MAE 10.23 · RMSE 14.12 · C-MAPSS skoru 45416.1 (örtüşen pencereler nedeniyle skor resmi protokolden daha büyük görünür — beklenen)

NOT: gerçek C-MAPSS, resmi test dosyası + RUL etiketleri (test RUL'u 125'te kırpılı). SQL öznitelikli model ile karşılaştırma: `sonuclar/faz2-rul-sql.md`.