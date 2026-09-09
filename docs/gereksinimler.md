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
| FR-02 | `dsp/features.py` (log-mel, zarf), `dsp/titresim.py` (order/envelope), `dsp/telemetri.py`; `02_oznitelik_cikar.py` | `test_dsp.py`, `test_titresim.py`, `test_telemetri.py`, `test_dsp_property.py`; C++ parite `test_parite.py` (Faz 4) | **kısmi** — Python tarafı tam, C++ paritesi Faz 4 |
| FR-03 | `models/autoencoder.py` (Faz 2) | `test_model_kalite.py::test_ae_auc` | planlandı |
| FR-04 | `models/cnn.py` (Faz 2) | `test_model_kalite.py::test_siniflandirma_f1` | planlandı |
| FR-05 | `models/rul.py` (Faz 2) | `test_model_kalite.py::test_rul_mae` | planlandı |
| FR-06 | `edge/onnx_disa.py` (Faz 3) | `test_runtime_parite.py` | planlandı |
| FR-07 | `protocol/mesaj.proto`, `protocol/yayinci.py` (Faz 5) | `test_protokol.py::test_round_trip` | planlandı |
| FR-08 | `serving/toplayici.py` (Faz 5) | `test_toplayici.py` | planlandı |
| FR-09 | `serving/dashboard/` (Faz 6) | uçtan uca senaryo | planlandı |
| PR-01..05 | — | `bench/` ölçüm betikleri | planlandı |
| RB-01 | `data/splits.py::rejim_holdout_split` | `test_splits.py::test_rejim_holdout_hedef_domain_test` | **kısmi** — split tarafı var |
| RB-02 | `eval/gorulmemiş_ariza.py` (Faz 2) | `test_model_kalite.py::test_gorulmemis_ariza` | planlandı |
| RB-03 | `edge/tampon.py` (Faz 5) | `test_protokol.py::test_ag_kesintisi` | planlandı |
| RB-04 | `edge/nicemleme.py` (Faz 3) | `test_nicemleme_parite.py` | planlandı |
| RB-05 | `dsp/dogrulama.py` (`sonlu_1b`, `pozitif`) — tüm DSP modülleri | `test_dsp.py::test_nan_girdi_hata`, `test_hizalama.py::test_kanal_nan_reddedilir`, `test_titresim.py::test_bant_gecir_gecersiz_bant` vb. | **karşılandı** |
| QR-01 | `.dvc/`, `data/splits.py::manifest_parmak_izi` | `test_splits.py::test_manifest_parmak_izi_kararli` | **karşılandı** |
| QR-02 | `.github/workflows/ci.yml` | — | **kısmi** — Faz 0 kapıları aktif |
| QR-03 | `pyproject.toml` cov ayarı | CI | **kısmi** |
| QR-04 | `sonuclar/saha-test-raporu.md` (Faz 6) | — | planlandı |
