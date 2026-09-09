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

- Şema round-trip: tespit → Protobuf → geri; şema evrimi (eski tüketici yeni mesaj).
- MQTT/DDS: uç → broker → merkez; mesaj sayısı, sıra, bozulma yok.
- **Ağ arıza enjeksiyonu** (`tc netem`): broker başta kapalı (tampon + retry), akış
  ortasında düşer (reconnect + backfill), yüksek gecikme, %kayıp, QoS 1 çift
  teslimat → tüketici idempotent (RB-03).
- Backpressure: merkez yavaş → uç tamponu dolar → zarif bozulma, çökme değil.
- Merkezi toplayıcı: 10k msg/s ingest kayıpsız; zaman-aralığı sorgusu doğru.
- **Uçtan uca**: sensör sim'ine enjekte anomali → dashboard'da N sn içinde doğru
  makine kimliği + şiddet.

Kod: `tests/test_protokol.py`, `tests/test_toplayici.py`

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
| 1 DSP birim | 1 | **aktif** — `test_dsp.py` + `test_hizalama.py` + `test_titresim.py` + `test_telemetri.py` + `test_dsp_property.py` (52 test); C++ golden-vector paritesi Faz 4 |
| 2 Model kalite | 2 | **aktif** — `test_metrics.py` + `test_baseline.py` + `test_autoencoder.py` + `test_oznitelik.py` (37 test); CNN/CRNN/Transformer + gerçek MIMII AUC hedefi sırada |
| 3 Domain shift | 2 | planlandı |
| 4 Uç/donanım | 3-4 | planlandı |
| 5 Protokol | 5 | planlandı |
| 6 CI kapıları | sürekli | **kısmi** — Faz 0 kapıları |
| 7 Saha kabul | 6 | planlandı |
