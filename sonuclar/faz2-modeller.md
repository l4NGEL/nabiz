# Faz 2 — model karşılaştırması

- öznitelik: a_ + z_ + t_  ·  gözetimsiz (yalnız normal eğitim)
- torch/AE: var  (hızlı mod: AE epok=8)
- NOT: sentetik veri — sayılar boru hattı doğrulaması içindir, portföy değeri için MIMII gerekir

| strateji      | model            |   AUC | AUC_CI         |   pAUC |    F1 |   precision |   recall |   FPR |   yanlış_alarm/sa |    sn |
|:--------------|:-----------------|------:|:---------------|-------:|------:|------------:|---------:|------:|------------------:|------:|
| birim_disi    | rms_esigi        | 0.042 | [0.001, 0.093] |  0.474 | 0.286 |       0.167 |    1     | 1     |           1800    | 0.675 |
| birim_disi    | izolasyon_ormani | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 0.931 |
| birim_disi    | mahalanobis      | 1     | [1.000, 1.000] |  1     | 0.915 |       1     |    0.844 | 0     |              0    | 0.709 |
| birim_disi    | dense_ae         | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 2.445 |
| rejim_holdout | rms_esigi        | 0     | [0.000, 0.000] |  0.474 | 0.255 |       0.149 |    0.875 | 1     |           1800    | 0.726 |
| rejim_holdout | izolasyon_ormani | 1     | [1.000, 1.000] |  1     | 0.877 |       0.78  |    1     | 0.056 |            101.25 | 1.086 |
| rejim_holdout | mahalanobis      | 1     | [1.000, 1.000] |  1     | 0.286 |       0.167 |    1     | 1     |           1800    | 0.77  |
| rejim_holdout | dense_ae         | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 1.066 |

## Makine tipine göre AUC kırılımı

- **birim_disi / rms_esigi** — fan: 0.032 (n=96), pump: 0.054 (n=96)
- **birim_disi / izolasyon_ormani** — fan: 1.000 (n=96), pump: 1.000 (n=96)
- **birim_disi / mahalanobis** — fan: 1.000 (n=96), pump: 1.000 (n=96)
- **birim_disi / dense_ae** — fan: 1.000 (n=96), pump: 1.000 (n=96)
- **rejim_holdout / rms_esigi** — fan: 0.000 (n=192), pump: 0.000 (n=192)
- **rejim_holdout / izolasyon_ormani** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / mahalanobis** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / dense_ae** — fan: 1.000 (n=192), pump: 1.000 (n=192)

## Domain-shift (RB-01)

- rms_esigi: birim_disi 0.042 → rejim_holdout 0.000  (Δ +0.042)
- izolasyon_ormani: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)
- mahalanobis: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)
- dense_ae: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)