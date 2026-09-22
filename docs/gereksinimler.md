# Gereksinimler ve İzlenebilirlik

**Belge no:** NABIZ-GRK-001 · **Sürüm:** 0.1 (taslak) · **Tarih:** 2026-09-08

Her gereksinim: kimlik · ifade · kabul kriteri · doğrulama yöntemi.
Doğrulama yöntemi kodları: **T** test (otomatik), **A** analiz/ölçüm, **D** demonstrasyon, **İ** inceleme.

---

## Fonksiyonel (FR)

| Kimlik | İfade | Kabul kriteri | Yöntem | Faz |
|--------|-------|---------------|--------|-----|
| FR-01 | Sistem akustik + titreşim + telemetri kanallarını alıp ortak zaman eksenine hizalar | Kayma < 1 ms; eksik kanal graceful ele alınır | T, A | 1 |
| FR-02 | Sistem her 2 sn'lik pencereden öznitelik çıkarır (log-mel, zarf/order spektrumu, telemetri istatistikleri) | Öznitelik deterministik; Python↔C++ parite tol. içinde | T | 1, 4 |
| FR-03 | Sistem gözetimsiz anomali skoru üretir (yalnız normalde eğitilmiş) | Test kümesinde AUC ≥ 0.90 (makine tipi ort.) | T, A | 2 |
| FR-04 | Sistem arıza tipini sınıflandırır (etiket varsa) | Makro-F1 ≥ 0.80; karışıklık matrisi raporlanır | T, A | 2 |
| FR-05 | Sistem kalan faydalı ömür (RUL) kestirir | C-MAPSS FD001'de MAE ≤ 15 çevrim | T, A | 2 |
| FR-06 | Model uç düğümde (ARM64 / x86) çıkarım yapar | ONNX Runtime + opsiyonel TensorRT; çıktı FP32 referansla tutarlı | T, A | 3 |
| FR-07 | Uç düğüm tespit mesajını tanımlı şema ile yayınlar | Protobuf mesajı IR-01'e uygun; MQTT/DDS'e publish | T, D | 5 |
| FR-08 | Merkezi toplayıcı mesajları alır, saklar, sorgulanabilir kılar | Zaman-aralığı sorgusu doğru satır kümesini döner | T | 5 |
| FR-09 | Dashboard canlı anomali skoru, alarm, karışıklık matrisi, gecikme gösterir | Enjekte anomali ≤ N sn içinde panelde görünür | D | 6 |

## Performans (PR)

| Kimlik | İfade | Kabul kriteri | Yöntem | Faz |
|--------|-------|---------------|--------|-----|
| PR-01 | Yanlış alarm oranı düşük | Nominal çalışmada ≤ 1 yanlış alarm / 24 saat (7 günlük kayıt) | A | 6 |
| PR-02 | Uçtan uca gecikme sınırlı | Hedef ARM64'te P95 (ses-giriş → tespit-çıkış) ≤ 500 ms; x86 tek çekirdek P95 ≤ 150 ms | A | 3 |
| PR-03 | Anomali ayrımı güçlü | pAUC (FPR ≤ 0.1) ≥ 0.85; DCASE dense-AE baseline'ını geçer | T, A | 2 |
| PR-04 | Uç modeli küçük | INT8 model ≤ 5 MB; RAM tepe ≤ 256 MB | A | 3 |
| PR-05 | Gerçek zamanlı işleme | Sürdürülebilir verim ≥ 1× (pencere/gerçek-süre) hedef donanımda | A | 3 |

## Arayüz (IR)

| Kimlik | İfade | Kabul kriteri | Yöntem | Faz |
|--------|-------|---------------|--------|-----|
| IR-01 | Tespit mesaj şeması sürümlü ve geriye uyumlu | Alanlar: zaman damgası, düğüm/makine kimliği, anomali skoru, sınıf, güven, şiddet, öznitelik anlık görüntüsü, model sürümü. `buf breaking` geçer | T, İ | 5 |
| IR-02 | Taşıma katmanı değiştirilebilir | MQTT birincil; DDS (Cyclone/Fast) adaptörü aynı arayüzü uygular | T, İ | 5 |
| IR-03 | Kayıt DB şeması tanımlı | Zaman serisi tablosu + retention/rollup; şema migration'lı | İ | 5 |

## Robüstluk (RB)

| Kimlik | İfade | Kabul kriteri | Yöntem | Faz |
|--------|-------|---------------|--------|-----|
| RB-01 | Domain-shift'e dayanıklı | Hedef domain AUC düşüşü kaynak domain'e göre ≤ 2 puan (MIMII DUE) | T, A | 2 |
| RB-02 | Görülmemiş arıza tipini tespit eder | {A,B} arızasında eğitilip C arızası verildiğinde anomali skoru eşiği geçer (recall ≥ 0.75) | T, A | 2 |
| RB-03 | Ağ kesintisinde veri kaybetmez | Broker 10 dk düşük → uç yerel tampon + reconnect'te backfill; kayıp = 0 | T, D | 5 |
| RB-04 | Nicemleme doğruluğu korur | INT8/FP16 AUC düşüşü ≤ 1 puan; örnek-bazı skor korelasyonu ≥ 0.99 | T, A | 3 |
| RB-05 | Bozulmuş girdiye çökmeden yanıt verir | NaN/inf, yanlış örnekleme hızı, sessizlik, kliplenme → tanımlı hata veya güvenli çıktı | T | 1 |

## Kalite / süreç (QR)

| Kimlik | İfade | Kabul kriteri | Yöntem | Faz |
|--------|-------|---------------|--------|-----|
| QR-01 | Kod ve veri versiyon kontrollü | Git + DVC; her bölme SHA-256 parmak izli; koşular parmak izi alıntılar | İ | 0 |
| QR-02 | CI regresyon kapıları | Her PR: birim + sızıntı + DSP golden-vector; model kalite kapısı; gecikme kapısı; şema uyum kapısı | T | 0, 2, 3, 5 |
| QR-03 | Test kapsamı | `src/nabiz` satır kapsamı ≥ %80 | T | sürekli |
| QR-04 | Saha test raporu | Her gereksinim → test senaryosu → sonuç; başarısızlık analizi; kısıtlar; PDF | İ, D | 6 |

---

## İzlenebilirlik matrisi

| Gereksinim | Tasarım / kod | Test senaryosu | Durum |
|------------|---------------|----------------|-------|
| FR-01 | `dsp/hizalama.py` — `hizala` (kayma < 1 ms assert), `telemetri_hizala`, eksik kanal graceful | `test_hizalama.py::*` (12 test) | **karşılandı** (gerçek eş-zamanlı kayıt Faz 6 saha) |
| FR-02 | `dsp/features.py` (log-mel, zarf), `dsp/titresim.py` (order/envelope), `dsp/telemetri.py`; `02_oznitelik_cikar.py` | `test_dsp.py`, `test_titresim.py`, `test_telemetri.py`, `test_dsp_property.py` | **kısmi** — Python tarafı tam; DSP'nin (WAV→mel) C++ portu YOK (bilinçli sınırlandırma, bkz. FR-06/Faz 4) — yalnız model-çıkarım C++'a taşındı |
| FR-03 | `models/`: baseline (RMS/IsoForest/Mahalanobis) + AE (dense) + CNN (ConvAE) + RNN (CRNNAE) + Transformer (SpekTransformerAE) + hibrid (FusionAE); `eval/metrics.py`; `03_model_egit.py` | `test_autoencoder.py`, `test_mel_modelleri.py`, `test_fusion.py`, `test_baseline.py`, `test_metrics.py` (~61 test) | **kısmi** — 7 model ailesi + harness gerçek MIMII'de çalıştı (ilan "CNN/RNN/LSTM/AE/Transformers/hibrid" maddesinin TAMAMI); AUC≥0.90 hedefi **karşılanmadı** — en iyi tip-içi rejim_holdout 0.80-0.86 (mahalanobis/dense_ae), birim_disi'de bazı modeller ≤0.5 (bkz. RB-01). Dürüstçe raporlandı, bkz. `sonuclar/faz2-modeller.md`. |
| FR-04 | `models/cnn.py` (Faz 2) | `test_model_kalite.py::test_siniflandirma_f1` | planlandı |
| FR-05 | `models/rul.py` (RULRegresor) + `data/cmapss.py` (gerçek+sentetik, resmi protokol); `05_rul_egit.py` | `test_cmapss.py`, `test_rul_model.py` (25 test) | **kısmi** — GRU baseline'ı (train-ortalaması) sentetikte açık farkla geçiyor (MAE 9.9 vs 76.1); gerçek C-MAPSS ile literatür karşılaştırması açık |
| FR-06 | `edge/onnx_disa.py` (export+parite) + `edge/nicemleme.py` (INT8); `04_uc_uyarla.py`; **`cpp/src/nabiz_cikarim.cpp`** (C++ ONNX Runtime çıkarımı, DenseAE) | `test_onnx_disa.py`, `test_nicemleme.py` (13 test) + `tests/test_cpp_parity.py` (3 test, build varsa) | **kısmi** — ONNX Runtime CPU (Python+C++) tam, golden-vector paritesi geçti; TensorRT/gerçek ARM/mel-tensör C++ desteği açık madde |
| FR-07 | `protocol/mesaj.proto` + `mesajlar.py` (oluştur/serileştir) + `yayinci.py` (MQTT publish + tampon) | `test_mesajlar.py` (7), `test_yayinci.py` (7) | **karşılandı** — DDS adaptörü hariç (yalnız MQTT) |
| FR-08 | `serving/toplayici.py` — FastAPI + SQLite, idempotent upsert, zaman/makine sorgusu | `test_toplayici.py` (9) | **karşılandı** — gerçek uçtan uca doğrulandı (12/12 HTTP ile alındı, `07_protokol_demo.py`) |
| FR-09 | `serving/dashboard/` (Faz 6) | uçtan uca senaryo | planlandı |
| IR-01 | `protocol/mesaj.proto` (11 alan) → `mesaj_pb2.py` (`grpc_tools.protoc`) | `test_mesajlar.py::test_serilestir_coz_roundtrip` | **kısmi** — şema + roundtrip tam; `buf breaking` CI kapısı yok |
| IR-02 | `protocol/yayinci.py` — `_MQTTIstemcisi` Protocol arayüzü (paho-mqtt somutlaştırması) | `test_yayinci.py` (7 test, enjekte sahte istemci) | **kısmi** — MQTT tam; DDS adaptörü yok |
| IR-03 | `serving/toplayici.py` — SQLite şema (`tespitler` tablosu, PK+indeks); sorgu arayüzü DB'den bağımsız tasarlandı | `test_toplayici.py` (9 test) | **kısmi** — SQLite'ta tam; TimescaleDB'ye geçiş + migration açık madde |
| PR-01 | `eval/metrics.py::yanlis_alarm_saat`; `03_model_egit.py` raporu | `test_metrics.py::test_yanlis_alarm_saat` | **kısmi** — gerçek MIMII'de ölçüldü: en iyi 14.2/saat (rms_esigi, rejim_holdout), hedef (≤1/24s) **karşılanmadı**. Eşik F1-optimal seçildi, yanlış-alarm-öncelikli eşikle (esik_sec hedef='fpr') iyileşebilir — açık madde. |
| PR-03 | `eval/metrics.py::pauc` (DCASE `max_fpr`) + baseline karşılaştırması | `test_metrics.py::test_pauc_*` | **kısmi** — gerçek MIMII'de en iyi pAUC 0.617 (rms_esigi, rejim_holdout) — hedef (≥0.85) **karşılanmadı**. DCASE dense-AE literatür baseline'ıyla sayısal karşılaştırma yapılmadı (açık madde); düşük-FPR bölgesinde (FPR≤0.1) genel olarak zayıf. |
| PR-02 | `edge/bench.py::gecikme_olc`; `04_uc_uyarla.py` raporu (x86 vekili) | `test_bench.py` | **kısmi** — x86 ölçümü var (P95 dense_ae 0.16 ms, conv_ae 1.6 ms sentetikte), gerçek ARM açık madde |
| PR-04, PR-05 | `04_uc_uyarla.py` (model boyutu); verim ölçümü açık | — | **kısmi** — boyut ölçülüyor (dense_ae INT8 0.07 MB), verim/RAM açık madde |
| RB-01 | `data/splits.py::rejim_holdout_split` + `03_model_egit.py` domain-shift Δ tablosu | `test_splits.py::test_rejim_holdout_*` | **kısmi** — SNR-holdout ile ölçüldü (MIMII DUE'nin resmi kaynak/hedef domain ayrımı DEĞİL, kendi vekilimiz); asıl bulgu domain-shift değil **cross-unit** (birim_disi) — MIMII'de o çok daha büyük bir düşüş yaratıyor (dense_ae'de -0.29 puan; ≤2 puan hedefinin çok üstünde). RB-01'in orijinal "≤2 puan" hedefi SNR-shift için de karşılanmadı (dense_ae Δ +0.29 — ama bu YÖN TERS: rejim_holdout birim_disi'den DAHA YÜKSEK AUC verdi, "düşüş" değil iyileşme). |
| RB-02 | `eval/gorulmemiş_ariza.py` (Faz 2) | `test_model_kalite.py::test_gorulmemis_ariza` | planlandı |
| RB-03 | `protocol/tampon.py` (yerel WAL, dosya-kalıcı) + `yayinci.py` (otomatik backfill) | `test_tampon.py` (6), `test_yayinci.py::test_baglanma_tetikler_backfill` | **kısmi karşılandı** — mantık tam test edilmiş (sahte istemci); gerçek broker'lı 10-dk kesinti senaryosu / `tc netem` paket-seviyesi test edilmedi |
| RB-04 | `edge/nicemleme.py::int8_nicemle` (MatMul/Gemm dinamik PTQ) + `edge/onnx_disa.py::skor_paritesi` | `test_nicemleme.py::test_int8_skor_paritesi_yuksek` | **kısmi** — dense_ae'de tam karşılandı (korr 1.0000, ΔAUC 0.000 sentetikte); ConvAE'de Conv katmanları nicemlenmiyor (bkz. yol-haritası) |
| RB-05 | `dsp/dogrulama.py` (`sonlu_1b`, `pozitif`) — tüm DSP modülleri | `test_dsp.py::test_nan_girdi_hata`, `test_hizalama.py::test_kanal_nan_reddedilir`, `test_titresim.py::test_bant_gecir_gecersiz_bant` vb. | **karşılandı** |
| QR-01 | `.dvc/`, `data/splits.py::manifest_parmak_izi` | `test_splits.py::test_manifest_parmak_izi_kararli` | **karşılandı** |
| QR-02 | `.github/workflows/ci.yml` | — | **kısmi** — Faz 0 kapıları aktif |
| QR-03 | `pyproject.toml` cov ayarı | CI | **kısmi** |
| QR-04 | `sonuclar/saha-test-raporu.md` (Faz 6) | — | planlandı |
