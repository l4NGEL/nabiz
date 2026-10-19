# Skor ailesi — validasyon seçimi

Eşik ve k yalnız validasyondan. Test sayıları rapor içindir.

| strateji | skor | val recall | uygun | z | k | test recall | FP olay | normal saat | PR-01 | dosya AUC | oturum recall |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| birim_disi | mahal/medyan | 0.518 | True | 2.0 | 6 | 0.557 | 0 | 11.40 | True | 0.925 | 1.000 |
| birim_disi | mahal/tepe3 | 0.555 | True | 1.5 | 6 | 0.613 | 0 | 11.40 | True | 0.901 | 1.000 |
| birim_disi | ledoit/medyan | 0.511 | True | 2.0 | 6 | 0.529 | 0 | 11.40 | True | 0.858 | 1.000 |
| birim_disi | ledoit/tepe3 | 0.544 | True | 1.5 | 6 | 0.570 | 2 | 11.40 | False | 0.821 | 1.000 |
| birim_disi | spe16/medyan | 0.485 | True | 2.0 | 6 | 0.552 | 0 | 11.40 | True | 0.848 | 1.000 |
| birim_disi | spe32/medyan | 0.518 | True | 2.0 | 6 | 0.569 | 0 | 11.40 | True | 0.870 | 1.000 |
| birim_disi | spe32/tepe3 | 0.484 | True | 2.0 | 6 | 0.538 | 0 | 11.40 | True | 0.834 | 1.000 |
| birim_disi | spe64/medyan | 0.563 | True | 2.0 | 6 | 0.631 | 0 | 11.40 | True | 0.916 | 1.000 |
| birim_disi | t2_32/medyan | 0.286 | True | 4.0 | 6 | 0.349 | 0 | 11.40 | True | 0.801 | 0.750 |
| birim_disi | gmm16x2/medyan | 0.476 | True | 2.0 | 6 | 0.420 | 1 | 11.40 | False | 0.825 | 1.000 |
| birim_disi | knn32/medyan | 0.556 | True | 1.5 | 6 | 0.593 | 1 | 11.40 | False | 0.835 | 1.000 |
| birim_disi | knn32/tepe3 | 0.551 | True | 1.5 | 6 | 0.582 | 4 | 11.40 | False | 0.798 | 1.000 |
| birim_disi | mahal+spe32 | 0.524 | True | 2.0 | 6 | 0.573 | 0 | 11.40 | True | 0.902 | 1.000 |
| birim_disi | mahal+knn | 0.552 | True | 1.5 | 6 | 0.611 | 0 | 11.40 | True | 0.890 | 1.000 |
| birim_disi | spe32+t2 | 0.449 | True | 2.5 | 6 | 0.471 | 0 | 11.40 | True | 0.842 | 1.000 |
| rejim_holdout | mahal/medyan | 0.536 | True | 1.5 | 3 | 0.743 | 8 | 21.73 | False | 0.927 | 1.000 |
| rejim_holdout | mahal/tepe3 | 0.494 | True | 1.5 | 6 | 0.709 | 2 | 21.73 | False | 0.914 | 1.000 |
| rejim_holdout | ledoit/medyan | 0.498 | True | 1.5 | 3 | 0.728 | 6 | 21.73 | False | 0.923 | 1.000 |
| rejim_holdout | ledoit/tepe3 | 0.452 | True | 1.5 | 6 | 0.689 | 1 | 21.73 | False | 0.907 | 1.000 |
| rejim_holdout | spe16/medyan | 0.475 | True | 2.0 | 6 | 0.694 | 1 | 21.73 | False | 0.913 | 1.000 |
| rejim_holdout | spe32/medyan | 0.567 | True | 1.5 | 6 | 0.778 | 1 | 21.73 | False | 0.919 | 1.000 |
| rejim_holdout | spe32/tepe3 | 0.529 | True | 1.5 | 6 | 0.734 | 1 | 21.73 | False | 0.901 | 1.000 |
| rejim_holdout | spe64/medyan | 0.646 | True | 1.5 | 6 | 0.845 | 2 | 21.73 | False | 0.934 | 1.000 |
| rejim_holdout | t2_32/medyan | 0.471 | True | 1.5 | 6 | 0.677 | 1 | 21.73 | False | 0.900 | 1.000 |
| rejim_holdout | gmm16x2/medyan | 0.445 | True | 1.5 | 6 | 0.677 | 1 | 21.73 | False | 0.916 | 1.000 |
| rejim_holdout | knn32/medyan | 0.513 | True | 1.5 | 6 | 0.711 | 0 | 21.73 | True | 0.918 | 1.000 |
| rejim_holdout | knn32/tepe3 | 0.498 | True | 1.5 | 6 | 0.672 | 0 | 21.73 | True | 0.897 | 1.000 |
| rejim_holdout | mahal+spe32 | 0.555 | True | 1.5 | 6 | 0.765 | 1 | 21.73 | False | 0.942 | 1.000 |
| rejim_holdout | mahal+knn | 0.529 | True | 1.5 | 6 | 0.713 | 0 | 21.73 | True | 0.942 | 1.000 |
| rejim_holdout | spe32+t2 | 0.525 | True | 1.5 | 6 | 0.734 | 1 | 21.73 | False | 0.924 | 1.000 |
