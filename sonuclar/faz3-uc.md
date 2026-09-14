# Faz 3 — uç uyarlama (ONNX / INT8 / gecikme)

Ölçüm ortamı: bu makinenin x86 CPU'su (ONNX Runtime `CPUExecutionProvider`).
Fiziksel uç donanım kullanılmadı — bkz. docs/yol-haritasi.md 'donanım gerekmez' notu.

| model | fp32 MB | int8 MB | ONNX parite (maks fark) | INT8 skor korr. | AUC fp32→int8 (Δ) | RB-04 | fp32 P50/P95 ms | int8 P50/P95 ms | PR-02 |
|---|---:|---:|---:|---:|---|:---:|---|---|:---:|
| dense_ae | 0.238 | 0.070 | 9.54e-07 | 1.0000 | 1.000→1.000 (+0.000) | ✓ | 0.062/0.164 | 0.120/0.181 | ✓ |
| conv_ae | 1.266 | 0.520 | 9.54e-07 | 1.0000 | 0.651→0.651 (+0.000) | ✓ | 0.553/1.596 | 0.560/2.018 | ✓ |

RB-04 kriteri: AUC düşüşü ≤ 0.01 VE skor korelasyonu ≥ 0.99. PR-02 kriteri (bu ölçümde x86 vekili): P95 ≤ 150 ms.
NOT: sentetik veride AUC zaten doygun (~1.0) — düşüş neredeyse sıfır görünür; gerçek MIMII'de RB-04 daha anlamlı bir test olur.