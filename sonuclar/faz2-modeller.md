# Faz 2 — model karşılaştırması

- öznitelik yolu: a_ + z_ + t_  ·  mel yolu: log-mel tensör (aktif)
- torch: var  (hızlı mod)
- gözetimsiz (yalnız normal eğitim); eşik validasyondan F1 ile
- NOT: sentetik veri — sayılar boru hattı doğrulaması içindir, portföy değeri için MIMII gerekir

| strateji      | model               |   AUC | AUC_CI         |   pAUC |    F1 |   precision |   recall |   FPR |   yanlış_alarm/sa |    sn |
|:--------------|:--------------------|------:|:---------------|-------:|------:|------------:|---------:|------:|------------------:|------:|
| birim_disi    | rms_esigi           | 0.042 | [0.001, 0.093] |  0.474 | 0.286 |       0.167 |    1     | 1     |           1800    | 0     |
| birim_disi    | izolasyon_ormani    | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 0.236 |
| birim_disi    | mahalanobis         | 1     | [1.000, 1.000] |  1     | 0.915 |       1     |    0.844 | 0     |              0    | 0.024 |
| birim_disi    | dense_ae            | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 1.406 |
| birim_disi    | conv_ae             | 0.651 | [0.530, 0.777] |  0.737 | 0.667 |       1     |    0.5   | 0     |              0    | 1.258 |
| birim_disi    | crnn_ae             | 0.977 | [0.942, 1.000] |  0.959 | 0.512 |       0.348 |    0.969 | 0.362 |            652.5  | 2.618 |
| birim_disi    | spek_transformer_ae | 0.998 | [0.993, 1.000] |  0.991 | 0.925 |       0.886 |    0.969 | 0.025 |             45    | 0.96  |
| birim_disi    | fusion_ae           | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 1.797 |
| rejim_holdout | rms_esigi           | 0     | [0.000, 0.000] |  0.474 | 0.255 |       0.149 |    0.875 | 1     |           1800    | 0     |
| rejim_holdout | izolasyon_ormani    | 1     | [1.000, 1.000] |  1     | 0.877 |       0.78  |    1     | 0.056 |            101.25 | 0.236 |
| rejim_holdout | mahalanobis         | 1     | [1.000, 1.000] |  1     | 0.286 |       0.167 |    1     | 1     |           1800    | 0.025 |
| rejim_holdout | dense_ae            | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 0.185 |
| rejim_holdout | conv_ae             | 1     | [1.000, 1.000] |  1     | 0.286 |       0.167 |    1     | 1     |           1800    | 1.122 |
| rejim_holdout | crnn_ae             | 1     | [1.000, 1.000] |  1     | 0.286 |       0.167 |    1     | 1     |           1800    | 2.464 |
| rejim_holdout | spek_transformer_ae | 1     | [1.000, 1.000] |  1     | 0.286 |       0.167 |    1     | 1     |           1800    | 0.821 |
| rejim_holdout | fusion_ae           | 1     | [1.000, 1.000] |  1     | 1     |       1     |    1     | 0     |              0    | 1.647 |

## Makine tipine göre AUC kırılımı

- **birim_disi / rms_esigi** — fan: 0.032 (n=96), pump: 0.054 (n=96)
- **birim_disi / izolasyon_ormani** — fan: 1.000 (n=96), pump: 1.000 (n=96)
- **birim_disi / mahalanobis** — fan: 1.000 (n=96), pump: 1.000 (n=96)
- **birim_disi / dense_ae** — fan: 1.000 (n=96), pump: 1.000 (n=96)
- **birim_disi / conv_ae** — fan: 0.685 (n=96), pump: 0.664 (n=96)
- **birim_disi / crnn_ae** — fan: 0.964 (n=96), pump: 0.993 (n=96)
- **birim_disi / spek_transformer_ae** — fan: 0.993 (n=96), pump: 1.000 (n=96)
- **birim_disi / fusion_ae** — fan: 1.000 (n=96), pump: 1.000 (n=96)
- **rejim_holdout / rms_esigi** — fan: 0.000 (n=192), pump: 0.000 (n=192)
- **rejim_holdout / izolasyon_ormani** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / mahalanobis** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / dense_ae** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / conv_ae** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / crnn_ae** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / spek_transformer_ae** — fan: 1.000 (n=192), pump: 1.000 (n=192)
- **rejim_holdout / fusion_ae** — fan: 1.000 (n=192), pump: 1.000 (n=192)

## Domain-shift (RB-01)

- rms_esigi: birim_disi 0.042 → rejim_holdout 0.000  (Δ +0.042)
- izolasyon_ormani: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)
- mahalanobis: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)
- dense_ae: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)
- conv_ae: birim_disi 0.651 → rejim_holdout 1.000  (Δ -0.349)
- crnn_ae: birim_disi 0.977 → rejim_holdout 1.000  (Δ -0.023)
- spek_transformer_ae: birim_disi 0.998 → rejim_holdout 1.000  (Δ -0.002)
- fusion_ae: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)

## Kalibrasyon + OOD (dense_ae) — RB benzeri sağlamlık kontrolü

ECE: ham (min-max normalize skor) vs Platt-kalibre olasılık, validasyon üzerinde uydurulmuş, test'te ölçülmüş. OOD AUC: eğitilmiş model, iç-dağılım normali gürültü/öznitelik-karışık (yapısı bozulmuş) girdiden ayırt edebiliyor mu (1.0 = tam ayırt ediyor, 0.5 = ayıramıyor → 'kendinden emin yanlış' riski).

| strateji | model | ECE ham | ECE kalibre | OOD AUC (gürültü) | OOD AUC (karışık) |
|---|---|---:|---:|---:|---:|
| birim_disi | dense_ae | 0.017 | 0.011 | 1.000 | 1.000 |
| rejim_holdout | dense_ae | 0.060 | 0.094 | 1.000 | 1.000 |