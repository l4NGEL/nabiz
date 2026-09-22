# Yol Haritası

**Sürüm:** 0.1 · **Tarih:** 2026-09-08 · Kaba süre: ~4-5 hafta yarı zamanlı
(~%45 mevcut projelerden yeniden kullanım).

Yeniden kullanım kaynakları:
- `sonar-hedef-siniflandirma` — mel/STFT, attention, kalibrasyon, OOD, çapraz-veri, ONNX, realtime
- `voice-recog` — ONNX/INT8, mobil hız ölçümü, sherpa-onnx, MLP kalibrasyon
- `AvioHMI` — JNI ile in-process C++/CUDA Sobel çekirdeği (CUDA deseni)
- `radar-tr-modul-phm` — FDC/FAR/RUL + FMEA + izlenebilirlik doküman üslubu

---

## Faz 0 — Kapsam + veri + iskele · 3-4 gün · **DEVAM EDİYOR**

- [x] Repo iskeleti, `pyproject.toml`, CI
- [x] `docs/veri-toplama-sartnamesi.md` — sensör/rejim/etiket/kalite/kaynak eşlemesi
- [x] `docs/gereksinimler.md` — FR/PR/IR/RB/QR + izlenebilirlik matrisi
- [x] `docs/test-plani.md` — 8 katman
- [x] `src/nabiz/data/splits.py` — birim-içi / birim-dışı / rejim-holdout + sızıntı kontrolü + parmak izi
- [x] `src/nabiz/data/manifest.py` — MIMII + sentetik manifest
- [x] `src/nabiz/dsp/features.py` — log-mel, çerçeveleme, zarf spektrumu (numpy/scipy)
- [x] `tests/` — split sızıntı + DSP bilinen-sinyal testleri
- [x] `00_veri_indir.py` — sentetik WAV üretimi (2 rejim) + MIMII indirme yardımı
- [x] `01_split_dogrula.py` — manifest → 3 split → sızıntı raporu + parmak izi
- [x] DVC init (`dvc init`) — `.dvc/` versiyonlu; remote + `dvc add` gerçek veri gelince
- [ ] MIMII fan/pump indir → `dvc add data/ham/mimii` → gerçek veri ile ilk koşu
- [ ] DVC remote (S3/GDrive/yerel) yapılandır

## Faz 1 — DSP önyüz · 3-4 gün · **TAMAM**

- [x] Çok-kanal hizalama (`dsp/hizalama.py`) — `Kanal`, `yeniden_ornekle`, `hizala`
  (ortak hız + ortak zaman aralığı, eksik kanal graceful), `capraz_korelasyon_kaymasi`,
  `telemetri_hizala` — FR-01
- [x] Titreşim analizi (`dsp/titresim.py`) — `rulman_ariza_frekanslari` (FTF/BPFO/BPFI/BSF),
  `bant_gecir`, `bantli_zarf_spektrumu`, `order_spektrumu` (sabit RPM),
  `hesaplanan_order_takibi` (COT, değişken hız), `spektral_kurtosis_bant`
- [x] Telemetri öznitelik çıkarımı (`dsp/telemetri.py`) — kanal başına 8 istatistik,
  `rejim_etiketi`, `sentetik_telemetri` (Faz 1 vekil)
- [x] `dsp/features.py` genişletildi — `zarf_oznitelik`, `pencere_oznitelikleri`
- [x] `02_oznitelik_cikar.py` — WAV → 2 sn pencere → 168 öznitelik/pencere
  (a_ 130 akustik, z_ 6 zarf, t_ 32 telemetri) → `data/islenmis/oznitelikler.parquet`
- [x] `src/nabiz/data/pipeline.py` — manifest+split üretimi `01`/`02` arası paylaşımlı
- [x] hypothesis property testleri (`tests/test_dsp_property.py`)
- [x] 68 test geçiyor, ruff temiz

Kalan (gerçek veriyle): titreşim yolunu MIMII yerine CWRU/Paderborn ile besle;
gerçek telemetri kaynağı (şu an sentetik vekil).

## Faz 2 — Modeller + değerlendirme · 1.5-2 hafta · **başladı**

- [x] `eval/metrics.py` — AUC, pAUC (DCASE `max_fpr`), eşik seçimi (F1/FPR hedefli),
  ECE (Guo), C-MAPSS asimetrik skor, `yanlis_alarm_saat`, `bootstrap_ci`, `anomali_raporu` (kırılımlı)
- [x] `models/baseline.py` — RMS eşiği, IsolationForest, Mahalanobis + `Olcekleyici`
- [x] `models/autoencoder.py` — dense AE (torch, GPU-farkında, erken durma, deterministik)
- [x] `data/oznitelik.py` — parquet → model-hazır diziler (split + grup/ablasyon seçimi)
- [x] `03_model_egit.py` — baseline + AE, birim_disi + rejim_holdout, `sonuclar/faz2-modeller.md`
  (AUC/pAUC/F1/FPR + makine tipi kırılımı + domain-shift Δ)
- [x] Bootstrap CI (`anomali_raporu` içinde)
- [x] `models/_egitim.py` — ortak AE eğitim çekirdeği (`TemelAE`, erken durma, deterministik)
- [x] `models/cnn.py` (ConvAE — 2B evrişimli), `models/crnn.py` (CRNNAE — GRU seq2seq),
  `models/transformer.py` (SpekTransformerAE — maskeli çerçeve yeniden yapılandırma) — log-mel tensörü
- [x] `dsp/features.py::mel_tensor` (sabit-boyut) + `02_oznitelik_cikar.py --mel` (`data/islenmis/mel.npz`)
  + `data/mel.py` yükleyici
- [x] `03_model_egit.py` — öznitelik yolu + mel yolu + füzyon, 7 model ailesi (CNN/RNN/AE/Transformer/hibrid)
- [x] `models/fusion.py` — `FusionAE`: spektrogram-CNN ⊕ telemetri-MLP, ortak darboğaz,
  ağırlıklı çift yeniden yapılandırma kaybı, dal-bazlı hata ayrıştırma (`anomali_skoru_dallar`)
- [x] `models/rul.py` — `RULRegresor` (GRU + MLP kafa, hedef standardizasyonu —
  standardize edilmemiş RUL'de ortalamaya sıkışma tespit edilip düzeltildi)
- [x] `data/cmapss.py` — gerçek format okuma (train/test/RUL dosyaları) + sentetik
  üretim (aynı şema) + `pencerele` (kaydırmalı, eğitim) + `son_pencere` (resmi
  protokol, birim başına 1 tahmin — C-MAPSS skorunu suni şişirmez)
- [x] `00_veri_indir.py --cmapss-sentetik` / `--cmapss-bilgi`
- [x] `05_rul_egit.py` — train→val/test birim split, GRU vs ortalama-baseline,
  MAE/RMSE/C-MAPSS skoru → `sonuclar/faz2-rul.md` (baseline'ı geçti: EVET)
- [x] Kalibrasyon (`eval/kalibrasyon.py::PlattKalibratoru` — val'de uydur, test'te ECE öncesi/sonrası)
  ve OOD (`eval/ood.py` — gürültü + öznitelik-karıştırma) `03_model_egit.py`'ye entegre
  (dense_ae için; ilginç bulgu: rejim_holdout'ta kalibrasyon domain-shift altında
  İYİLEŞMEDİ [ECE 0.060→0.094] — val kaynak domaininden, test hedeften; dürüstçe raporlandı)
- [ ] Ablasyon (a_/z_/t_ grupları) + 3-5 seed
- [ ] MLflow deney takibi
- [x] **Gerçek MIMII ile yeniden çalıştır** (2026-09-21/22) — fan+pump, 6dB+0dB, 19.510
  dosya, 175.590 pencere, TAM mod, iki strateji → `sonuclar/faz2-modeller.md`. **Ana bulgu:**
  birim_disi (görülmemiş ünite) rejim_holdout'tan (görülmemiş SNR) sistematik daha zor
  (her modelde +Δ, dense_ae +0.29/mahalanobis +0.26 puan), bazı modeller birim_disi'de
  rastgeleden kötü — epok-bağımsız (hızlı=8 vs tam=120 epok aynı), gerçek cross-machine
  genelleme zorluğu (DCASE "unseen machine ID" ile tutarlı), eğitim eksikliği DEĞİL.
  Tip-içi rejim_holdout AUC 0.80-0.86 (mahalanobis/dense_ae) — havuzlanmış 0.66'dan
  çok daha güçlü (iki tip farklı skor ölçeğinde). Yol boyunca kritik bir sentetik-telemetri
  etiket sızıntısı bulunup düzeltildi (commit 775e6ad) — FusionAE'nin sahte AUC 1.000'i.
- [x] Domain-shift (birim_disi vs rejim_holdout, gerçek MIMII) — yukarıdaki madde kapsıyor;
  MIMII DUE'nin resmi kaynak/hedef domain ayrımı henüz kullanılmadı (kendi SNR-holdout'umuz var)
- [ ] Görülmemiş arıza tipi, çapraz-veri (CWRU→Paderborn) — test katmanı 3, hâlâ açık

## Faz 3 — Uç uyarlama + optimizasyon · 1 hafta · **başladı**

- [x] `edge/onnx_disa.py` — ONNX export (dinamik yığın ekseni), FP32 parite doğrulama
  (maks fark, korelasyon), anomali-skoru paritesi. Çok-girdili modeller de (FusionAE) destekli.
  `dynamo=False` kasıtlı (legacy TorchScript exporter — kod içinde gerekçeli)
- [x] `edge/nicemleme.py` — PTQ INT8 dinamik nicemleme; **öntanım yalnız MatMul/Gemm**
  (Conv2d dinamik nicemleme ONNX Runtime CPU EP'de `ConvInteger` NOT_IMPLEMENTED —
  gerçek/belgelenen kısıt, statik+kalibrasyonlu nicemleme gerekir → açık madde)
- [x] `edge/bench.py` — tek-örnek gecikme P50/P95/P99 + verim (ONNX Runtime CPU)
- [x] `04_uc_uyarla.py` — DenseAE + ConvAE: export → nicemle → parite → gecikme →
  `sonuclar/faz3-uc.md` (RB-04 + PR-02 kabul kriteri sütunları)
- [x] Nicemleme paritesi testleri — RB-04 (`test_nicemleme.py`)
- [x] Gecikme ölçüm testleri — PR-02 (`test_bench.py`)
- [ ] QAT (quantization-aware training) — şu an yalnız PTQ
- [ ] Conv2d için statik nicemleme (kalibrasyon verisiyle) — ConvAE'de INT8 şu an yalnız Linear katmanları kapsıyor
- [ ] Budama + distilasyon → küçük öğrenci model
- [ ] TensorRT motor (RTX 2080 lokal; hedef Jetson/Ampere) — şu an yalnız ONNX Runtime CPU EP
- [ ] Gerçek ARM ölçümü: Oracle Cloud Always Free / QEMU (donanım gerekmez — bkz. sohbet geçmişi)
- [ ] Soak testi kurulumu (24-72 saat, bellek sızıntısı)

## Faz 4 — C++ çıkarım yolu · 4-5 gün · **başladı**

- [x] `cpp/` — CMakeLists.txt + `src/nabiz_cikarim.cpp`: ONNX Runtime C++ API ile
  öznitelik-vektörü (DenseAE, 168d) çıkarımı. MSVC/VS2022 + CMake 3.31 ile ilk denemede derlendi.
- [x] ONNX Runtime C++ SDK indirme + link (`third_party/onnxruntime/`, git dışı) — README'de belgelendi
- [x] Golden-vector parite testi (Python ↔ C++) — test katmanı 1/4 (`06_golden_uret.py` +
  `cpp/testler/dogrula.py` + `tests/test_cpp_parity.py`, CMake build yoksa otomatik `skip`).
  **FP32: maks fark 4.5e-6, korelasyon 1.000000 (tam parite).**
  **INT8: korelasyon 1.000000, maks fark ~0.05 — Python wheel vs C++ SDK derlemesi arası
  kuantize çekirdek farkı, belgelenen/beklenen bulgu, sıralama bozulmuyor.**
- [x] C++ gecikme ölçümü: P50 ≈ 0.02 ms (Python ONNX Runtime'dan hızlı — binding yükü yok)
- [ ] Sensör akışından (WAV) uçtan uca: şu an C++ yalnız ÖZNİTELİK VEKTÖRÜ alıyor;
  ham WAV→mel çıkarımı C++'a taşınmadı (büyük ek iş — DSP'nin tamamının C++ portu).
  Bilinçli sınırlandırma, `cpp/src/nabiz_cikarim.cpp` başlığında gerekçeli.
- [ ] Mel-tensör modelleri (ConvAE/CRNN/Transformer, 3B girdi) C++'ta desteklenmiyor — yalnız DenseAE
- [ ] CUDA mel çekirdeği (AvioHMI Sobel deseni) — opsiyonel, RTX 2080'de mümkün
- [ ] CI'da derleme (CI runner'da MSVC/CMake yok — GitHub Actions `windows-latest` + CMake ile mümkün, henüz eklenmedi)

## Faz 5 — Protokol + merkez · 4-5 gün · **büyük ölçüde tamam**

- [x] `protocol/mesaj.proto` — IR-01 (mesaj_id, zaman damgası, düğüm/makine kimliği,
  anomali skoru, sınıf, güven, şiddet, öznitelik anlık görüntüsü, model sürümü);
  `grpc_tools.protoc` ile üretildi (`mesaj_pb2.py`, commit'li)
- [x] `protocol/mesajlar.py` — oluştur/serileştir/çöz/sözlük dönüşümü + alan doğrulama
- [x] `protocol/tampon.py` — yerel WAL (SQLite, MQTT'ye bağımsız, dosya-kalıcı) — RB-03
- [x] `protocol/yayinci.py` — MQTT yayıncı (paho-mqtt 2.x), tampon + otomatik backfill
  (`on_connect`'te), enjekte edilebilir istemci (broker'sız tam birim testi) — IR-02
- [x] `serving/toplayici.py` — FastAPI + SQLite, `mesaj_id` PK ile idempotent upsert
  (QoS1 çift teslim güvenli), zaman/makine sorgusu, istatistik — FR-08, IR-03
- [x] `07_protokol_demo.py` — sentetik tespit → yayınla → toplayıcıya gönder →
  `sonuclar/faz5-protokol.md`. **Gerçek uçtan uca doğrulandı**: uvicorn toplayıcı
  başlatılıp 12/12 tespit HTTP ile alındı, `/istatistik` doğru agregasyon verdi.
  Broker'sız MQTT davranışı da doğrulandı: 12/12 mesaj tampona düştü (RB-03).
- [x] `docker-compose.yml` — mosquitto + toplayıcı (Docker Desktop bu oturumda kapalı,
  derleme denenmedi ama yapılandırma standart/gözden geçirildi)
- [x] tests: `test_mesajlar.py`, `test_tampon.py`, `test_yayinci.py` (sahte istemci,
  broker'sız tam mantık), `test_toplayici.py` (FastAPI TestClient) — 29 test
- [ ] DDS adaptörü (Cyclone/Fast) — yalnız MQTT var
- [ ] `tc netem` ağ arıza enjeksiyonu — Linux-özel; Windows'ta broker durdur/başlat ile
  eşdeğer davranış (`docker-compose stop mosquitto`) test edildi ama gerçek `tc netem`
  paket-seviyesi testi değil
- [ ] TimescaleDB'ye geçiş (şu an SQLite — üretim için belgelenen yükseltme yolu)
- [ ] `buf` lint + breaking kapısı (CI'da proto uyum kontrolü)

## Faz 6 — Saha testi + rapor + dashboard · 1 hafta

- [ ] `serving/dashboard/` — canlı skor, alarm, karışıklık matrisi, gecikme (sonar/CV-matcher tek-sayfa deseni)
- [ ] Saha testi protokolü: görülmemiş makine/arıza, enjekte gürültü, domain shift
- [ ] `sonuclar/saha-test-raporu.md` → PDF — QR-04
- [ ] 10 dk demo senaryosu + ekran kaydı
- [ ] README rozetleri, sürüm etiketi v0.1

---

## Portföy / CV bağlantısı

Tamamlandığında `CV_Aselsan_*.tex` proje satırı: çok-sensör (titreşim/akustik/telemetri)
füzyon, gözetimsiz + gözetimli + RUL, ONNX/INT8/TensorRT uç uyarlama, C++/CUDA çıkarım,
Protobuf/MQTT/DDS merkez haberleşme, saha test raporu. İlan maddeleri karşılığı:
`docs/gereksinimler.md` izlenebilirlik matrisi.
