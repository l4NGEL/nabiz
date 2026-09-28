# Faz 2 — SQL tabanlı RUL hattı (gerçek C-MAPSS)

- veri: NASA C-MAPSS (PCoE), `FD001` modellendi; DB: **postgresql**
- hacim: **265,256** ölçüm satırı, 8 kaynak (FD001–FD004 train+test)
- sensörler (train motorlarından, std > 0.01): 14 — s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
- öznitelik: 85 (SQL pencere fonksiyonları: ort10/30, eğim10/30, var30, sapma)
- bölme: 79 train / 21 val motor (motor bazında); resmi test: 100 motor, motor başına son çevrim

## Hacim ve sorgu performansı

| kaynak | satır |
|---|---:|
| FD001-test | 13,096 |
| FD001-train | 20,631 |
| FD002-test | 33,991 |
| FD002-train | 53,759 |
| FD003-test | 16,596 |
| FD003-train | 24,720 |
| FD004-test | 41,214 |
| FD004-train | 61,249 |

- öznitelik sorgusu, `FD001-train` (20,631 satır): 2.76 sn (satırlar Python'a çekilerek, 3 denemenin en iyisi)
- tüm tablo (265,256 satır): 36.24 sn; `EXPLAIN ANALYZE` sunucu yürütme süresi 16548 ms
- parite (SQL vs bağımsız pandas, 20,631 satır × 84 öznitelik): maks göreli fark 4.0e-08, NaN deseni aynı

NOT: Bu hacim küçük-orta ölçektir (tek makine, ~265 bin satır); "büyük veri" iddiası değildir. Sayı olduğu gibi raporlanır, veri çoğaltılmamıştır.

## Model seçimi (yalnız val motorları, rastgele kesilme noktası)

| model | RMSE | MAE |
|---|---:|---:|
| ort_taban | 42.58 | 37.30 |
| hgb_sig | 8.78 | 7.14 |
| hgb_orta ← seçildi | 8.43 | 6.62 |
| hgb_derin | 9.49 | 7.23 |

## Resmi FD001 testi (100 motor, bir kez)

| model | RMSE | MAE | C-MAPSS skoru (düşük iyi) |
|---|---:|---:|---:|
| hgb_orta (SQL öznitelikleri) | 11.37 | 8.30 | 195.7 |
| ort_taban (train ortalaması) | 41.94 | 34.83 | 33354.5 |

Test RUL'u 125'te kırpılmıştır (literatürde yaygın). Kırpılmamış gerçek RUL'a karşı: RMSE 12.47, MAE 9.36 (11 motorun gerçek RUL'u 125'in üstünde).

Sızıntı kontrolleri: eğitim etiketleri karıştırılınca test RMSE **41.21** (taban 41.94); sensörsüz, yalnız `cevrim` ile **31.22** — kazanım sensör özniteliklerinden geliyor. Test öznitelikleri yalnız kesik test kayıtlarından, nedensel pencerelerle hesaplanır (bkz. `tests/test_sql_oznitelik.py`).

## Bakım önceliklendirme

Risk eşikleri (tahmini RUL, çevrim): KRITIK ≤ 30, YUKSEK ≤ 60, IZLE ≤ 90, aksi NORMAL — planlama varsayımı, veri gereksinimi değil.

- Spearman (tahmin vs gerçek RUL sırası): **0.907**
- gerçek RUL ≤ 30 olan 25 motorun KRITIK/YUKSEK işaretlenme oranı: **100%**
- en riskli 10 tahminden gerçekten kritik (RUL ≤ 30) olan: **100%**

Seviye karışıklığı (satır gerçek, sütun tahmin):

| gerçek \ tahmin | KRITIK | YUKSEK | IZLE | NORMAL |
|---|---:|---:|---:|---:|
| KRITIK | 23 | 2 | 0 | 0 |
| YUKSEK | 2 | 11 | 1 | 0 |
| IZLE | 0 | 2 | 5 | 8 |
| NORMAL | 0 | 0 | 5 | 41 |

İlk 10 (en riskli):

| sıra | motor | tahmini RUL | risk | gerçek RUL |
|---:|---:|---:|---|---:|
| 1 | 76 | 5.5 | KRITIK | 10 |
| 2 | 81 | 5.8 | KRITIK | 8 |
| 3 | 68 | 6.8 | KRITIK | 8 |
| 4 | 34 | 7.3 | KRITIK | 7 |
| 5 | 31 | 7.6 | KRITIK | 8 |
| 6 | 82 | 8.2 | KRITIK | 9 |
| 7 | 35 | 11.3 | KRITIK | 11 |
| 8 | 42 | 12.2 | KRITIK | 10 |
| 9 | 20 | 15.2 | KRITIK | 16 |
| 10 | 49 | 16.3 | KRITIK | 21 |

## Kapsam ve sınırlar

- C-MAPSS **simüle edilmiş** motor verisidir (NASA); gerçek filo verisi değildir. Sonuçlar bir havacılık bakım senaryosu prototipidir.
- Yalnız FD001 (tek çalışma koşulu, tek arıza modu) modellendi; FD002/FD004 çoklu çalışma koşulu için koşul-bazlı normalizasyon gerektirir (yapılmadı).
- Seçim yalnız val'de yapıldı; test sonucuna bakılarak hiçbir ayar değiştirilmedi. Tek tohum — belirsizlik aralığı raporlanmadı.
- Aynı resmi testte ham 21 sensör penceresiyle GRU: `sonuclar/faz2-rul.md`.
