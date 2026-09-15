# Test Planı

**Belge no:** NABIZ-TST-001 · **Sürüm:** 0.1 · **Tarih:** 2026-09-08

"Test" bu sistemde 8 ayrı katman demek. ASELSAN UGES ilan profili bunlardan
ikisini açıkça istiyor: *"başarım ve performans ölçümü"* (katman 4) ve
*"saha testlerinde denenmesi ... raporlamak"* (katman 7).

Ayrı bir "test fazı" yok — her katman ilgili faza gömülü. Kaba efor: proje
süresinin ~%25-30'u.

---

## 0. Veri ve split doğrulama — Faz 0, CI'da her PR

Sensör-ML'de sessiz hata kaynağı #1: veri sızıntısı.

- **Makine-birimi ayrık split** — bir `makine_birimi` hem train hem test'te ise CI başarısız.
- Çalışma rejimi kapsamı: test, train rejimlerini kapsıyor mu (ya da bilinçli holdout).
- Etiket sağlığı: sınıf dengesi, hash ile duplike dosya, sessiz/klipli segment.
- Run-to-failure veri: zamansal split — asla shuffle.
- DVC hash sabitleme: test seti donmuş; parmak izi eşleşmezse başarısız.

Kod: `tests/test_splits.py` · `src/nabiz/data/splits.py`

## 1. DSP önyüz birim testleri — Faz 1

- Determinizm: aynı girdi → bit-aynı çıktı.
- Bilinen sinyal: `f` Hz sinüs → doğru mel bandında tepe; beyaz gürültü → düz-ish
  spektrum; AM sinyal → zarf spektrumunda modülasyon frekansında tepe; Parseval.
- Kenar durumlar: DC offset, kliplenme, NaN/inf, yanlış örnekleme hızı, çok kısa
  çerçeve, sessizlik (RB-05).
- **Python ↔ C++/CUDA parite**: golden-vector testi — sabit girdi seti + repoda
  tutulan beklenen diziler; tolerans içinde uyuşmalı (Faz 4).
- Property-based (hypothesis): rastgele geçerli sinyal → çökme yok, şekil değişmez.

Kod: `tests/test_dsp.py` · (Faz 4) `tests/test_parite.py`

## 2. Model kalite değerlendirmesi — Faz 2

MIMII/MIMII DUE = DCASE Task 2; onun metodolojisi.

- Birincil: **AUC + pAUC** (FPR ≤ 0.1), makine-tipi ve birim kırılımı.
- Anomali (gözetimsiz): validasyonda eşik seç → precision/recall/F1/yanlış alarm.
- Sınıflandırma: karışıklık matrisi, makro-F1, sınıf-bazı recall.
- RUL: MAE, RMSE + C-MAPSS asimetrik skor (geç tahmin ağır cezalı).
- Kalibrasyon: ECE + güvenilirlik diyagramı, temperature scaling öncesi/sonrası.
- OOD/belirsizlik: yabancı makine tipi / saf gürültü → yüksek belirsizlik, kendinden
  emin yanlış değil.
- **Yenilecek baseline'lar**: çoğunluk sınıfı, IsolationForest (el öznitelikleri),
  DCASE dense-AE, RMS eşiği. Derin modeller geçmezse **dürüstçe raporlanır**.
- Ablasyon: her modalite tek / füzyon; telemetri koşullama var/yok; mimariler aynı
  split'te kafa kafaya.
- İstatistik: 3-5 seed, ortalama ± std; "füzyon > akustik-tek mi" → bootstrap CI.

Kod: `tests/test_model_kalite.py` · `src/nabiz/eval/`

## 3. Domain shift / robüstluk — Faz 2 sonu

- **MIMII DUE**: kaynak vs hedef domain AUC farkı (RB-01).
- Enjekte bozulmalar: SNR (−6/0/+6 dB), örnekleme kayması, kazanç kayması, eksik
  telemetri kanalı, akışta paket kaybı → metrik–şiddet eğrisi.
- **Görülmemiş arıza tipi** (RB-02): {A,B}'de eğit, C'yi tespit et — gözetimsiz AE
  hâlâ işaretliyor mu? AE seçiminin gerekçesi bu testte kanıtlanır.
- Çapraz-veri: CWRU'da eğit → Paderborn'da test.

Kod: `tests/test_robustluk.py` · `src/nabiz/eval/bozulma.py`

## 4. Uç / donanım testleri — Faz 3-4 ("başarım ve performans ölçümü")

- **Gecikme**: uçtan uca P50/P95/P99, gerçek hedefte, N ≥ 1000, sıcak+soğuk;
  bütçeye assert (PR-02).
- Verim (PR-05), bellek/model boyutu (PR-04), güç (USB ölçer varsa: mJ/çıkarım).
- **Nicemleme paritesi** (RB-04): INT8/FP16 vs FP32 → AUC düşüşü ≤ 1 puan, skor
  korelasyonu ≥ 0.99.
- Runtime paritesi: PyTorch vs ONNX Runtime vs TensorRT, maks mutlak fark loglanır.
- **Soak testi**: uçta 24-72 saat kesintisiz → RSS düz (bellek sızıntısı yok),
  çökme yok, sabit girişte tespit oranı sabit.
- Termal: sürekli çıkarımda throttle / gecikme.

Not: fiziksel donanım gerekmez — x86 lokal + Oracle Cloud Always Free ARM VM +
QEMU emülasyon + Colab/Kaggle T4 (TensorRT) ile 0 TL yapılabilir.

Kod: `bench/` · `tests/test_runtime_parite.py`

## 5. Protokol / entegrasyon testleri — Faz 5

- [x] Şema round-trip: tespit → Protobuf → geri (`test_mesajlar.py`).
- [x] Yerel tampon/backfill mantığı: broker başta kapalı (tampon'a yazar), `on_connect`
  tetiklenince backfill (`test_yayinci.py` — enjekte sahte istemci, broker'sız).
  **Gerçek broker'la da doğrulandı** (bu oturumda): broker yokken 12/12 mesaj
  tampona düştü — beklenen RB-03 davranışı, `07_protokol_demo.py` çıktısı.
- [x] Merkezi toplayıcı: idempotent upsert (`mesaj_id` PK, QoS1 çift teslim güvenli),
  zaman-aralığı/makine sorgusu, istatistik agregasyonu (`test_toplayici.py`).
  **Gerçek uçtan uca**: uvicorn başlatıldı, 12/12 tespit HTTP ile alındı,
  `/istatistik` 3 makine için doğru ort/maks skor + adet döndü.
- [ ] MQTT/DDS canlı broker üzerinden: mosquitto/Docker bu oturumda çalışmadı
  (Docker Desktop kapalı) — `docker-compose.yml` hazır, `docker compose up` ile
  denenmedi. DDS adaptörü yok.
- [ ] **Ağ arıza enjeksiyonu** (`tc netem`): Linux-özel, bu Windows makinede
  doğrudan yapılamadı. Eşdeğeri (broker durdur/başlat) tasarımda var, canlı
  denenmedi. Yüksek gecikme/%kayıp/QoS1 çift teslim senaryoları açık.
- [ ] Backpressure (merkez yavaş → uç tamponu dolar), 10k msg/s ingest — açık.
- [ ] **Uçtan uca dashboard**: Faz 6, henüz yok.

Kod: `tests/test_mesajlar.py`, `tests/test_tampon.py`, `tests/test_yayinci.py`, `tests/test_toplayici.py`

## 6. CI / regresyon kapıları — sürekli

- Her PR: birim + entegrasyon + DSP golden-vector + split-sızıntı.
- **Model kalite kapısı**: donmuş küçük alt kümede hızlı eval; baseline'a göre AUC
  eşikten fazla düşerse başarısız.
- **Gecikme kapısı**: CI runner'da mikro-benchmark; çıkarım süresi %20+ geriler ise
  başarısız.
- Şema uyum kapısı: `buf breaking`.
- Gecelik: tam eval sweep + domain-shift takımı → metrikler MLflow + trend paneli.

Kod: `.github/workflows/`

## 7. Saha kabul testi — yazılı teslimat (Faz 6)

`docs/gereksinimler.md` izlenebilirlik matrisinin aynı üslubu: her gereksinim →
test senaryosu → geç/kal kriteri → sonuç.

Örnek kabul kriterleri (tam liste `sonuclar/saha-test-raporu.md`):

| Gereksinim | Kriter |
|------------|--------|
| FR-03 | Enjekte dış-bilezik rulman arızası, SNR ≥ 0 dB'de 60 sn içinde ≥ %90 tespit |
| PR-01 | Nominal çalışmada 7 günlük kayıtta ≤ 1 yanlış alarm / 24 saat |
| PR-02 | Hedef ARM64'te P95 uçtan-uca ≤ 500 ms |
| RB-03 | Broker 10 dk düşük → 0 mesaj kaybı, reconnect'te backfill |
| RB-04 | Hedef domain AUC düşüşü kaynak domain'e göre ≤ 2 puan |

Çıktılar:
- **Saha test raporu** (PDF): yöntem, test ortamı, veri kaynağı, sonuç tablosu,
  başarısızlık analizi (ne kaçtı, neden), kısıtlar, öneriler.
- **Demo senaryosu**: 10 dk scripted — uç düğümü başlat, sağlıklı→bozulan→arıza
  geçişli kayıt oynat, dashboard yansın, merkezi DB kaydı + RUL geri sayımı.
  Portföy ekran kaydı bu.

---

## Katman → faz → durum

| Katman | Faz | Durum |
|--------|-----|-------|
| 0 Veri/split | 0 | **aktif** — `test_splits.py` (16 test) |
| 1 DSP birim | 1 | **aktif** — `test_dsp.py` + `test_hizalama.py` + `test_titresim.py` + `test_telemetri.py` + `test_dsp_property.py` (52 test); DSP'nin (mel/order) C++ portu YOK — yalnız model-çıkarımın C++ paritesi geçti (`test_cpp_parity.py`, katman 4'te) |
| 2 Model kalite | 2 | **aktif** — `test_metrics.py` + `test_baseline.py` + `test_autoencoder.py` + `test_mel_modelleri.py` + `test_fusion.py` + `test_oznitelik.py` + `test_mel_loader.py` + `test_cmapss.py` + `test_rul_model.py` + `test_kalibrasyon.py` + `test_ood.py` (~102 test); kalibrasyon (Platt/ECE) + OOD (gürültü/karıştırma AUC) `03_model_egit.py`'ye entegre; gerçek MIMII/C-MAPSS AUC/MAE hedefi kalan tek açık madde |
| 3 Domain shift | 2 | **kısmi** — `03_model_egit.py` domain-shift Δ tablosu (birim_disi vs rejim_holdout); MIMII DUE + görülmemiş arıza tipi testleri sırada |
| 4 Uç/donanım | 3-4 | **aktif** — `test_onnx_disa.py` + `test_nicemleme.py` + `test_bench.py` + `test_cpp_parity.py` (20 test); x86 CPU'da gerçek ölçüm (`sonuclar/faz3-uc.md`), C++ golden-vector paritesi geçti (FP32 tam, INT8 gevşek); TensorRT/gerçek ARM/soak sırada |
| 5 Protokol | 5 | **büyük ölçüde tamam** — `test_mesajlar.py` + `test_tampon.py` + `test_yayinci.py` + `test_toplayici.py` (29 test); gerçek uçtan uca (HTTP toplayıcı) bu oturumda doğrulandı; canlı MQTT broker/DDS/`tc netem` sırada |
| 6 CI kapıları | sürekli | **kısmi** — Faz 0 kapıları |
| 7 Saha kabul | 6 | planlandı |
