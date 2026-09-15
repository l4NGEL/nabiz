# Faz 2 — model karşılaştırması

- öznitelik yolu: a_ + z_ + t_  ·  mel yolu: log-mel tensör (aktif)
- torch: var  (hızlı mod)
- gözetimsiz (yalnız normal eğitim); eşik validasyondan F1 ile
- NOT: sentetik veri — sayılar boru hattı doğrulaması içindir, portföy değeri için MIMII gerekir

| strateji      | model               |   AUC | AUC_CI         |   pAUC |    F1 |   precision |   recall |   FPR |   yanlış_alarm/sa |      sn |
|:--------------|:--------------------|------:|:---------------|-------:|------:|------------:|---------:|------:|------------------:|--------:|
| birim_disi    | rms_esigi           | 0.726 | [0.714, 0.737] |  0.671 | 0.181 |       0.099 |    1     | 1     |          1799.6   |   0.034 |
| birim_disi    | izolasyon_ormani    | 0.955 | [0.952, 0.959] |  0.837 | 0.433 |       0.896 |    0.286 | 0.004 |             6.567 |   0.907 |
| birim_disi    | mahalanobis         | 0.949 | [0.945, 0.953] |  0.812 | 0.337 |       0.203 |    0.994 | 0.431 |           775.124 |   0.544 |
| birim_disi    | dense_ae            | 0.968 | [0.964, 0.972] |  0.891 | 0.772 |       0.803 |    0.743 | 0.02  |            36.318 |  13.582 |
| birim_disi    | conv_ae             | 0.543 | [0.529, 0.556] |  0.512 | 0.181 |       0.099 |    1     | 1     |          1800     | 214.118 |
| birim_disi    | crnn_ae             | 0.475 | [0.460, 0.493] |  0.573 | 0.181 |       0.099 |    1     | 1     |          1800     | 605.695 |
| birim_disi    | spek_transformer_ae | 0.594 | [0.579, 0.611] |  0.638 | 0.179 |       0.099 |    0.947 | 0.957 |          1722.29  | 228.365 |
| birim_disi    | fusion_ae           | 1     | [1.000, 1.000] |  1     | 1     |       0.999 |    1     | 0     |             0.1   | 412.991 |
| rejim_holdout | rms_esigi           | 0.664 | [0.652, 0.674] |  0.666 | 0.403 |       0.99  |    0.253 | 0     |             0.587 |   0.034 |
| rejim_holdout | izolasyon_ormani    | 0.795 | [0.789, 0.802] |  0.603 | 0.31  |       0.186 |    0.933 | 0.497 |           895.172 |   0.91  |
| rejim_holdout | mahalanobis         | 0.997 | [0.996, 0.997] |  0.983 | 0.924 |       0.881 |    0.971 | 0.016 |            28.701 |   0.589 |
| rejim_holdout | dense_ae            | 0.989 | [0.988, 0.990] |  0.952 | 0.803 |       0.698 |    0.946 | 0.05  |            89.624 |   7.455 |
| rejim_holdout | conv_ae             | 0.58  | [0.572, 0.589] |  0.538 | 0.17  |       0.107 |    0.418 | 0.426 |           766.818 | 126.081 |
| rejim_holdout | crnn_ae             | 0.601 | [0.590, 0.611] |  0.594 | 0.276 |       0.233 |    0.338 | 0.135 |           242.998 | 197.45  |
| rejim_holdout | spek_transformer_ae | 0.769 | [0.760, 0.778] |  0.662 | 0.425 |       0.439 |    0.413 | 0.064 |           115.284 |  82.39  |
| rejim_holdout | fusion_ae           | 1     | [1.000, 1.000] |  1     | 0.998 |       0.996 |    1     | 0     |             0.854 | 152.839 |

## Makine tipine göre AUC kırılımı

- **birim_disi / rms_esigi** — pump: 0.726 (n=20088)
- **birim_disi / izolasyon_ormani** — pump: 0.955 (n=20088)
- **birim_disi / mahalanobis** — pump: 0.949 (n=20088)
- **birim_disi / dense_ae** — pump: 0.968 (n=20088)
- **birim_disi / conv_ae** — pump: 0.543 (n=20088)
- **birim_disi / crnn_ae** — pump: 0.475 (n=20088)
- **birim_disi / spek_transformer_ae** — pump: 0.594 (n=20088)
- **birim_disi / fusion_ae** — pump: 1.000 (n=20088)
- **rejim_holdout / rms_esigi** — pump: 0.664 (n=37845)
- **rejim_holdout / izolasyon_ormani** — pump: 0.795 (n=37845)
- **rejim_holdout / mahalanobis** — pump: 0.997 (n=37845)
- **rejim_holdout / dense_ae** — pump: 0.989 (n=37845)
- **rejim_holdout / conv_ae** — pump: 0.580 (n=37845)
- **rejim_holdout / crnn_ae** — pump: 0.601 (n=37845)
- **rejim_holdout / spek_transformer_ae** — pump: 0.769 (n=37845)
- **rejim_holdout / fusion_ae** — pump: 1.000 (n=37845)

## Domain-shift (RB-01)

- rms_esigi: birim_disi 0.726 → rejim_holdout 0.664  (Δ +0.062)
- izolasyon_ormani: birim_disi 0.955 → rejim_holdout 0.795  (Δ +0.160)
- mahalanobis: birim_disi 0.949 → rejim_holdout 0.997  (Δ -0.048)
- dense_ae: birim_disi 0.968 → rejim_holdout 0.989  (Δ -0.021)
- conv_ae: birim_disi 0.543 → rejim_holdout 0.580  (Δ -0.038)
- crnn_ae: birim_disi 0.475 → rejim_holdout 0.601  (Δ -0.126)
- spek_transformer_ae: birim_disi 0.594 → rejim_holdout 0.769  (Δ -0.175)
- fusion_ae: birim_disi 1.000 → rejim_holdout 1.000  (Δ +0.000)

## Kalibrasyon + OOD (dense_ae) — RB benzeri sağlamlık kontrolü

ECE: ham (min-max normalize skor) vs Platt-kalibre olasılık, validasyon üzerinde uydurulmuş, test'te ölçülmüş. OOD AUC: eğitilmiş model, iç-dağılım normali gürültü/öznitelik-karışık (yapısı bozulmuş) girdiden ayırt edebiliyor mu (1.0 = tam ayırt ediyor, 0.5 = ayıramıyor → 'kendinden emin yanlış' riski).

| strateji | model | ECE ham | ECE kalibre | OOD AUC (gürültü) | OOD AUC (karışık) |
|---|---|---:|---:|---:|---:|
| birim_disi | dense_ae | 0.099 | 0.090 | 0.995 | 0.975 |
| rejim_holdout | dense_ae | 0.105 | 0.023 | 0.999 | 0.999 |