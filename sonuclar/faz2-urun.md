# Faz 2 — ürün noktası

Üç kurulum, aynı Mahalanobis çekirdeği:
- **havuz** — tüm tipler tek model (önceki raporun tabanı)
- **tip** — fan ve pompa ayrı ayrı eğitilir; eşik tipin kendi validasyonundan
- **tip_uyarlama** — her ünite medyan/MAD ile kendi taban çizgisine oturtulur (hedef bölmede etiketsiz). Görülmemiş ünitenin 'normal'i filodan uzaksa skor onu arıza sanmasın diye.

İşletme noktaları (tip başına eşik, hatalar toplanır):
- `f1` — validasyonda F1
- `fpr01` — FPR ≈ 0.001 (2 sn pencerede ≈ 1.8 yanlış alarm/saat)
- `pr01` — PR-01 bütçesi: ≤ 1 yanlış alarm / 24 saat (FPR ≈ 2.3e-5)

Telemetri (`t_`) sentetik vekildir; ablasyon onu akustikten ayırır.

## Karşılaştırma

| strateji      | kurulum      | grup   |   makro_auc |   havuz_auc |   makro_pauc |   f1_recall |   f1_alarm_sa |   fpr01_recall |   fpr01_alarm_sa |   pr01_recall |   pr01_alarm_sa |
|:--------------|:-------------|:-------|------------:|------------:|-------------:|------------:|--------------:|---------------:|-----------------:|--------------:|----------------:|
| birim_disi    | havuz        | a+z+t  |       0.517 |       0.398 |        0.529 |       0.369 |       606.579 |          0.005 |            9.747 |             0 |               0 |
| birim_disi    | tip          | a+z+t  |       0.475 |       0.373 |        0.507 |       0.574 |      1198.05  |          0.013 |           63.694 |             0 |               0 |
| birim_disi    | tip          | a      |       0.483 |       0.386 |        0.514 |       0.581 |      1199.95  |          0.009 |           64.474 |             0 |               0 |
| birim_disi    | tip          | z      |       0.43  |       0.471 |        0.554 |       0.847 |      1171.3   |          0.052 |           35.185 |             0 |               0 |
| birim_disi    | tip          | t      |       0.5   |       0.496 |        0.499 |       0.996 |      1778.31  |          0.001 |            1.559 |             0 |               0 |
| birim_disi    | tip          | a+z    |       0.476 |       0.374 |        0.507 |       0.537 |      1173.49  |          0.013 |           64.766 |             0 |               0 |
| birim_disi    | tip_uyarlama | a+z+t  |       0.898 |       0.91  |        0.752 |       0.867 |       423.489 |          0.135 |            8.041 |             0 |               0 |
| birim_disi    | tip_uyarlama | a      |       0.913 |       0.923 |        0.75  |       0.949 |       632.992 |          0.133 |           15.643 |             0 |               0 |
| birim_disi    | tip_uyarlama | z      |       0.694 |       0.71  |        0.569 |       0.496 |       257.359 |          0.003 |            1.608 |             0 |               0 |
| birim_disi    | tip_uyarlama | t      |       0.501 |       0.58  |        0.5   |       0.972 |      1717.64  |          0     |            0     |             0 |               0 |
| birim_disi    | tip_uyarlama | a+z    |       0.905 |       0.917 |        0.755 |       0.924 |       569.591 |          0.136 |           10.478 |             0 |               0 |
| rejim_holdout | havuz        | a+z+t  |       0.832 |       0.661 |        0.709 |       0.873 |       649.233 |          0.007 |            0.511 |             0 |               0 |
| rejim_holdout | tip          | a+z+t  |       0.878 |       0.832 |        0.748 |       0.877 |       493.839 |          0.012 |            0.511 |             0 |               0 |
| rejim_holdout | tip          | a      |       0.877 |       0.832 |        0.737 |       0.787 |       326.431 |          0.008 |            0.486 |             0 |               0 |
| rejim_holdout | tip          | z      |       0.608 |       0.551 |        0.578 |       0.826 |       973.569 |          0.05  |            1.764 |             0 |               0 |
| rejim_holdout | tip          | t      |       0.499 |       0.498 |        0.5   |       0.962 |      1649.9   |          0.001 |            1.84  |             0 |               0 |
| rejim_holdout | tip          | a+z    |       0.88  |       0.834 |        0.749 |       0.833 |       393.149 |          0.012 |            0.511 |             0 |               0 |
| rejim_holdout | tip_uyarlama | a+z+t  |       0.921 |       0.908 |        0.811 |       0.806 |       177.94  |          0.235 |            1.866 |             0 |               0 |
| rejim_holdout | tip_uyarlama | a      |       0.92  |       0.916 |        0.811 |       0.808 |       170.859 |          0.2   |            1.943 |             0 |               0 |
| rejim_holdout | tip_uyarlama | z      |       0.849 |       0.816 |        0.686 |       0.756 |       360.327 |          0.103 |           22.06  |             0 |               0 |
| rejim_holdout | tip_uyarlama | t      |       0.499 |       0.471 |        0.5   |       0.863 |      1312.19  |          0     |            0     |             0 |               0 |
| rejim_holdout | tip_uyarlama | a+z    |       0.925 |       0.914 |        0.815 |       0.835 |       224.949 |          0.24  |            1.994 |             0 |               0 |

## Tip kırılımı

- **birim_disi / havuz / a+z+t** — fan 0.493 / pump 0.541 (makro 0.517)
- **birim_disi / tip_uyarlama / a+z+t** — fan 0.896 / pump 0.899 (makro 0.898)
- **birim_disi / tip_uyarlama / a** — fan 0.913 / pump 0.913 (makro 0.913)
- **birim_disi / tip_uyarlama / a+z** — fan 0.907 / pump 0.902 (makro 0.905)
- **rejim_holdout / havuz / a+z+t** — fan 0.803 / pump 0.861 (makro 0.832)
- **rejim_holdout / tip_uyarlama / a+z+t** — fan 0.903 / pump 0.939 (makro 0.921)
- **rejim_holdout / tip_uyarlama / a** — fan 0.906 / pump 0.934 (makro 0.920)
- **rejim_holdout / tip_uyarlama / a+z** — fan 0.906 / pump 0.944 (makro 0.925)

## Okuma

Ünite uyarlaması, havuzlanmış modelin görülmemiş ünitede çökmesini (havuz AUC ~0.40) tip-içi AUC ~0.90 bandına taşır. Telemetri (`t_`) her iki stratejide de AUC ~0.50 — sentetik kanal sinyal taşımıyor, füzyona katılmamalı. Zarflı akustik (`a+z`) SNR kaymasında en iyi; görülmemiş ünitede salt akustik (`a`) biraz daha yüksek.
- `birim_disi` akustik + uyarlama: makro AUC 0.913, pAUC 0.750, FPR≈0.001 iken recall 0.133 (15.64 yanlış alarm/saat), PR-01 recall 0.000.
- `rejim_holdout` akustik + uyarlama: makro AUC 0.920, pAUC 0.811, FPR≈0.001 iken recall 0.200 (1.94 yanlış alarm/saat), PR-01 recall 0.000.

Bu koşuda PR-01 (≤ 1 yanlış alarm / 24 saat) recall en fazla 0.000. En yüksek makro pAUC 0.815 (hedef 0.85). Sıralama düzeldi; düşük-FPR kuyruğu bu bütçeyi taşımıyor. Uyarlama etiketsizdir; istatistik, skorlanan partinin kendi pencerelerinden (medyan/MAD) gelir — tek pencereyi filo prototipine vurmak değildir. Pencere başına eşik PR-01'i tutmaz; alarm kararı dosya düzeyindedir.

## Dosya alarmı (PR-01)

Karar birimi dosyadır (~10 sn). Skor, pencerelerin medyanı veya p90'ıdır. Alarm, aynı kayıt klasöründe k ardışık dosya eşiği geçince bir kez sayılır. Eşik ham skor değil, her ünitenin kendi dosya medyan/MAD'ine göre z'dir; böylece başka ünitede seçilen eşik görülmemiş makineye taşınır. k, validasyonda ≤ 1 olay / 24 saat bütçesini tutanlar arasından recall'u yüksek ve en kısa olandır.

| strateji      | grup   | yontem   |   k | secildi   |   makro_auc |   recall |   olay_fp |   yanlis_alarm_saat |   normal_saat | pr01   |   val_recall |
|:--------------|:-------|:---------|----:|:----------|------------:|---------:|----------:|--------------------:|--------------:|:-------|-------------:|
| birim_disi    | a      | medyan   |   6 | True      |      0.9253 |   0.5575 |         0 |               0     |       11.4    | True   |       0.5183 |
| birim_disi    | a      | p90      |   6 | False     |      0.8977 |   0.4978 |         0 |               0     |       11.4    | True   |       0.4939 |
| birim_disi    | a+z    | medyan   |   6 | True      |      0.9161 |   0.5662 |         0 |               0     |       11.4    | True   |       0.5071 |
| birim_disi    | a+z    | p90      |   6 | False     |      0.8892 |   0.5163 |         0 |               0     |       11.4    | True   |       0.4827 |
| rejim_holdout | a      | medyan   |   3 | True      |      0.9271 |   0.6753 |         3 |               0.138 |       21.7333 | False  |       0.4297 |
| rejim_holdout | a      | p90      |   6 | False     |      0.9128 |   0.6577 |         1 |               0.046 |       21.7333 | False  |       0.4259 |
| rejim_holdout | a+z    | medyan   |   6 | True      |      0.9336 |   0.6665 |         1 |               0.046 |       21.7333 | False  |       0.4259 |
| rejim_holdout | a+z    | p90      |   6 | False     |      0.9192 |   0.6484 |         1 |               0.046 |       21.7333 | False  |       0.4221 |

Seçilenler:

- **birim_disi / a / medyan / k=6** — dosya makro AUC 0.925 (fan 0.922 / pump 0.928), recall 0.557, 0 olay / 11.40 saat normal (0.0000/saat). PR-01 geçti.
- **birim_disi / a+z / medyan / k=6** — dosya makro AUC 0.916 (fan 0.918 / pump 0.915), recall 0.566, 0 olay / 11.40 saat normal (0.0000/saat). PR-01 geçti.
- **rejim_holdout / a / medyan / k=3** — dosya makro AUC 0.927 (fan 0.912 / pump 0.942), recall 0.675, 3 olay / 21.73 saat normal (0.1380/saat). PR-01 kaldı.
- **rejim_holdout / a+z / medyan / k=6** — dosya makro AUC 0.934 (fan 0.912 / pump 0.955), recall 0.666, 1 olay / 21.73 saat normal (0.0460/saat). PR-01 kaldı.
