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

## Faz 1 — DSP önyüz · 3-4 gün

- [ ] Çok-kanal hizalama (`dsp/hizalama.py`) — FR-01
- [ ] Titreşim için order/envelope analizi parametreli
- [ ] Telemetri öznitelik çıkarımı (RPM/yük/sıcaklık istatistikleri, trend)
- [ ] Öznitelik hattını `02_oznitelik_cikar.py` ile diske yaz (parquet)
- [ ] hypothesis property testleri

## Faz 2 — Modeller + değerlendirme · 1.5-2 hafta

- [ ] `models/autoencoder.py` (dense + conv) — DCASE baseline eşdeğeri
- [ ] `models/cnn.py`, `models/crnn.py`, `models/transformer.py` (küçük AST/PaSST)
- [ ] `models/fusion.py` — spektrogram-CNN ⊕ telemetri-MLP
- [ ] `models/rul.py` — C-MAPSS
- [ ] `eval/metrics.py` — AUC, pAUC, ECE, C-MAPSS skoru
- [ ] Kalibrasyon (sonar `10_vliz_calibrate` uyarlaması), OOD (sonar `15_vliz_ood`)
- [ ] Baseline'lar: IsolationForest, RMS eşiği, DCASE dense-AE
- [ ] Ablasyon + 3-5 seed + bootstrap CI
- [ ] MLflow deney takibi
- [ ] Domain-shift (MIMII DUE), görülmemiş arıza tipi, çapraz-veri — test katmanı 3

## Faz 3 — Uç uyarlama + optimizasyon · 1 hafta

- [ ] `edge/onnx_disa.py` — ONNX export + doğrulama (sonar `03_onnx_disa_aktar` uyarlaması)
- [ ] PTQ INT8 + QAT; nicemleme paritesi testi — RB-04
- [ ] Budama + distilasyon → küçük öğrenci model
- [ ] TensorRT motor (RTX 2080 lokal; hedef Jetson/Ampere)
- [ ] `bench/` — gecikme P50/P95/P99, verim, bellek, güç; x86 + Oracle ARM + QEMU
- [ ] Soak testi kurulumu

## Faz 4 — C++ çıkarım yolu · 4-5 gün

- [ ] `cpp/` — ONNX Runtime C++ API çıkarım ikili dosyası: sensör akışı → öznitelik → model → tespit
- [ ] CUDA mel çekirdeği (AvioHMI Sobel deseni) — opsiyonel, RTX 2080'de
- [ ] Golden-vector parite testi (Python ↔ C++) — test katmanı 1
- [ ] CMake + CI'da derleme

## Faz 5 — Protokol + merkez · 4-5 gün

- [ ] `protocol/mesaj.proto` — IR-01; `buf` lint + breaking kapısı
- [ ] `protocol/yayinci.py` — MQTT (mosquitto); DDS adaptörü (Cyclone) — IR-02
- [ ] `edge/tampon.py` — yerel WAL + backfill — RB-03
- [ ] `serving/toplayici.py` — FastAPI + TimescaleDB/SQLite — FR-08, IR-03
- [ ] `tc netem` ağ arıza enjeksiyon testleri — test katmanı 5
- [ ] docker-compose: uç sim + broker + toplayıcı + DB

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
