# Resmi ızgara — z ≥ 2, eşit recall'da en uzun k

Seçim yalnız validasyon. Test sütunları rapor içindir.

| strateji | skor | val recall | uygun | z | k | test recall | FP olay | normal saat | PR-01 | dosya AUC | oturum recall |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| birim_disi | mahal | 0.518 | True | 2.0 | 6 | 0.557 | 0 | 11.40 | True | 0.925 | 1.000 |
| birim_disi | spe64 | 0.563 | True | 2.0 | 6 | 0.631 | 0 | 11.40 | True | 0.916 | 1.000 |
| birim_disi | spe96 | 0.574 | True | 2.0 | 6 | 0.677 | 0 | 11.40 | True | 0.945 | 1.000 |
| birim_disi | spe120 | 0.428 | True | 4.0 | 6 | 0.369 | 0 | 11.40 | True | 0.889 | 0.500 |
| birim_disi | kuyruk32 | 0.566 | True | 2.0 | 6 | 0.640 | 0 | 11.40 | True | 0.938 | 1.000 |
| birim_disi | kuyruk64 | 0.547 | True | 2.5 | 6 | 0.605 | 0 | 11.40 | True | 0.947 | 1.000 |
| birim_disi | kuyruk96 | 0.534 | True | 2.5 | 6 | 0.578 | 0 | 11.40 | True | 0.944 | 1.000 |
| birim_disi | knn32 | 0.501 | True | 2.0 | 6 | 0.521 | 1 | 11.40 | False | 0.835 | 1.000 |
| birim_disi | mahal+spe64 | 0.539 | True | 2.0 | 6 | 0.604 | 0 | 11.40 | True | 0.924 | 1.000 |
| birim_disi | mahal+knn | 0.510 | True | 2.0 | 6 | 0.535 | 0 | 11.40 | True | 0.890 | 1.000 |
| birim_disi | spe64+knn | 0.531 | True | 2.0 | 6 | 0.574 | 0 | 11.40 | True | 0.890 | 1.000 |
| birim_disi | max(mahal,spe64) | 0.548 | True | 2.0 | 6 | 0.607 | 0 | 11.40 | True | 0.917 | 1.000 |
| birim_disi | max(spe64,knn) | 0.533 | True | 2.0 | 6 | 0.586 | 0 | 11.40 | True | 0.889 | 1.000 |
| rejim_holdout | mahal | 0.430 | True | 2.0 | 6 | 0.675 | 1 | 21.73 | False | 0.927 | 1.000 |
| rejim_holdout | spe64 | 0.593 | True | 2.0 | 6 | 0.759 | 0 | 21.73 | True | 0.934 | 1.000 |
| rejim_holdout | spe96 | 0.544 | True | 2.0 | 6 | 0.788 | 1 | 21.73 | False | 0.914 | 1.000 |
| rejim_holdout | spe120 | 0.540 | True | 2.0 | 6 | 0.706 | 2 | 21.73 | False | 0.832 | 1.000 |
| rejim_holdout | kuyruk32 | 0.570 | True | 2.0 | 6 | 0.770 | 2 | 21.73 | False | 0.923 | 1.000 |
| rejim_holdout | kuyruk64 | 0.597 | True | 2.0 | 6 | 0.805 | 3 | 21.73 | False | 0.921 | 1.000 |
| rejim_holdout | kuyruk96 | 0.536 | True | 2.0 | 6 | 0.797 | 3 | 21.73 | False | 0.903 | 1.000 |
| rejim_holdout | knn32 | 0.437 | True | 2.0 | 6 | 0.602 | 0 | 21.73 | True | 0.918 | 1.000 |
| rejim_holdout | mahal+spe64 | 0.544 | True | 2.0 | 6 | 0.729 | 2 | 21.73 | False | 0.957 | 1.000 |
| rejim_holdout | mahal+knn | 0.456 | True | 2.0 | 6 | 0.655 | 0 | 21.73 | True | 0.942 | 1.000 |
| rejim_holdout | spe64+knn | 0.532 | True | 2.0 | 6 | 0.696 | 0 | 21.73 | True | 0.952 | 1.000 |
| rejim_holdout | max(mahal,spe64) | 0.559 | True | 2.0 | 6 | 0.737 | 1 | 21.73 | False | 0.958 | 1.000 |
| rejim_holdout | max(spe64,knn) | 0.555 | True | 2.0 | 6 | 0.714 | 0 | 21.73 | True | 0.955 | 1.000 |

## Ürün kararı — SPE-70

Bileşen sayısı aynı kuralda (z ≥ 2, validasyon recall) tarandı. 72 ve 80 validasyonda biraz daha yüksek recall verdi; SNR testinde 21.7 saatte 1 yanlış olay üretti. 70, iki kabul testinde de 0 yanlış olay bırakan en yüksek tespit.

| strateji | skor | val recall | z | k | test recall | FP olay | PR-01 | dosya AUC |
|---|---|---:|---:|---:|---:|---:|---|---:|
| birim_disi | spe70 | 0.585 | 2.0 | 6 | 0.654 | 0 | geçti | 0.927 |
| rejim_holdout | spe70 | 0.597 | 2.0 | 6 | 0.787 | 0 | geçti | 0.933 |

Mahalanobis taban (aynı kural): görülmemiş ünitede recall 0.557 / 0 olay, SNR kaymasında recall 0.675 / 1 olay. Canlı yol `AkustikUrun` bu SPE-70 noktasını kullanır. Eşik test etiketine bakılarak seçilmedi.
