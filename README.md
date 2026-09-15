# NABIZ

**Titreşim + akustik + telemetri ile teçhizat durum izleme ve anomali tespiti.**
Uç düğümde C++/CUDA çıkarım, tespitleri gerçek protokolle merkezi kayıt/izleme
platformuna akıtan uçtan uca sistem, savunma üslubu gereksinim takibi + saha test raporu.

Portföy projesi — ASELSAN UGES "Yapay Zeka (ve Algoritma) Geliştirme Mühendisi"
ilan profiline göre kurgulandı: sensör verisi → YZ modeli → uç/merkez dağıtım →
saha testi → raporlama.

## Durum

| Faz | Kapsam | Durum |
|-----|--------|-------|
| 0 | Kapsam + veri şartnamesi + split doğrulama + CI | **tamam** (kalan: DVC remote + gerçek MIMII) |
| 1 | DSP önyüz — hizalama, order/envelope, telemetri, öznitelik hattı | **tamam** (68 test) |
| 2 | Değerlendirme (AUC/pAUC/ECE) + 7 model ailesi + RUL (C-MAPSS, GRU) + kalibrasyon/OOD | **büyük ölçüde tamam** (177 test); ablasyon/MLflow + gerçek MIMII/C-MAPSS sırada |
| 3 | ONNX export + INT8 (PTQ) + gecikme ölçümü (x86 CPU) | **başladı**; TensorRT + Conv statik nicemleme + gerçek ARM sırada |
| 4 | C++ ONNX Runtime çıkarım yolu | **başladı** (DenseAE; golden-vector parite GEÇTİ); mel-tensör modelleri + CUDA çekirdeği sırada |
| 5 | Protobuf + MQTT + merkezi toplayıcı (FastAPI/SQLite) | **büyük ölçüde tamam** — gerçek uçtan uca doğrulandı; DDS/`tc netem`/canlı broker sırada |
| 6 | Saha testi simülasyonu + rapor + dashboard | planlandı |

Ayrıntı: [docs/yol-haritasi.md](docs/yol-haritasi.md)

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  Linux/mac: source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[dl]"        # Faz 2+ : torch (autoencoder) + onnx/onnxruntime (uç uyarlama)
                              # CUDA için torch'u ayrıca kur:
                              #   pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`torch` yoksa `03_model_egit.py` yalnız baseline'ları çalıştırır (AE atlanır); `onnx`/`onnxruntime`
yoksa `04_uc_uyarla.py` hata mesajıyla çıkar. `pytest` eksik bağımlılığa göre ilgili testleri `skip` eder.

## Hızlı başlangıç (gerçek veri olmadan)

```bash
python 00_veri_indir.py --sentetik      # data/ham/sentetik/ altına küçük WAV seti (2 rejim)
python 01_split_dogrula.py               # manifest + 3 split + sızıntı kontrolü + parmak izi
python 02_oznitelik_cikar.py --mel       # WAV → 2 sn pencere → 168 öznitelik + log-mel tensör
python 03_model_egit.py --hizli          # 7 model ailesi → sonuclar/faz2-modeller.md
python 04_uc_uyarla.py --hizli           # ONNX + INT8 + gecikme → sonuclar/faz3-uc.md
python 00_veri_indir.py --cmapss-sentetik && python 05_rul_egit.py --hizli  # RUL → sonuclar/faz2-rul.md
pytest -q                                # 206 birim testi (C++ build varsa +3 = 209)
```

## Faz 4 — C++ ONNX Runtime çıkarım yolu

Kapsam: DenseAE (öznitelik-vektörü, 168-boyut) — WAV→mel çıkarımı henüz C++'ta
değil (Python `dsp/features.py`'de), bu bilinçli bir sınırlama (bkz. `cpp/src/nabiz_cikarim.cpp`
başlığı). ConvAE/CRNN/Transformer (mel-tensör, 3B girdi) henüz desteklenmiyor.

**Gereksinimler:** CMake ≥3.20 + MSVC (Visual Studio "Desktop development with C++"
bileşeni; VS2022 kendi CMake'ini de getirir) + ONNX Runtime C++ SDK.

```bash
# 1) ONNX Runtime C++ SDK indir (~80 MB), third_party/onnxruntime/ altına çıkar (git dışı)
curl -L -o third_party/onnxruntime.zip \
  https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip
# third_party/ içinde unzip et, klasörü third_party/onnxruntime/ olarak adlandır

# 2) CMake configure + build (VS2022 kendi CMake'i; PATH'te yoksa tam yol kullan)
cmake -G "Visual Studio 17 2022" -A x64 -S cpp -B cpp/build
cmake --build cpp/build --config Release
# çıktı: cpp/build/Release/nabiz_cikarim.exe (+ onnxruntime.dll yanına kopyalanır)

# 3) Golden-vector parite testi (Python ONNX Runtime vs C++ ONNX Runtime, aynı .onnx)
python 04_uc_uyarla.py --hizli            # modeller/edge/dense_ae_fp32.onnx üretir
python 06_golden_uret.py                  # cpp/testler/altin_{girdi,skor}.csv üretir
python cpp/testler/dogrula.py             # C++ ikilisini çalıştırır, karşılaştırır
```

**Sonuç (bu makinede):** FP32 — maks fark 4.5e-6, korelasyon 1.000000 (tam parite).
INT8 — korelasyon 1.000000 ama maks fark ~0.05 (sıkı toleransı aşıyor): Python
onnxruntime wheel'i ile C++ SDK derlemesi arasında kuantize çekirdeklerde küçük
sayısal fark var — **beklenen, belgelenen bir bulgu**, sıralama/ayrım bozulmuyor.
C++ çıkarım gecikmesi P50 ≈ 0.02 ms (Python ONNX Runtime'dan da hızlı — dil
binding'i ek yükü yok).

`pytest` bu testleri (`tests/test_cpp_parity.py`) CMake build yoksa otomatik atlar.

## Faz 5 — Protokol + merkezi toplayıcı

```bash
pip install -e ".[protokol]"   # fastapi, uvicorn, httpx, paho-mqtt, grpcio-tools

# Toplayıcıyı başlat (ayrı terminal)
uvicorn nabiz.serving.toplayici:app --app-dir src --port 8700

# Demo: sentetik tespit üret, MQTT'ye yayınla (broker yoksa tamponlanır — RB-03),
# toplayıcıya HTTP ile gönder
python 07_protokol_demo.py
```

Bu makinede Docker çalışmadığı için MQTT broker'sız denendi: 12/12 mesaj beklendiği
gibi yerel tampona düştü. Toplayıcı ayrıca başlatılıp gerçek HTTP ile test edildi:
**12/12 tespit alındı**, `/istatistik` 3 makine için doğru ortalama/maks skor
döndürdü. Canlı MQTT (broker + backfill) için: `docker compose up`, sonra
`docker compose stop mosquitto` ile RB-03'ü canlı gözlemle (bkz. `docker-compose.yml`
başlığı).

Mesaj şeması: [src/nabiz/protocol/mesaj.proto](src/nabiz/protocol/mesaj.proto)

## Gerçek veri

MIMII (akustik endüstriyel makine — DCASE Task 2 tabanı), CWRU / Paderborn rulman
(titreşim), NASA C-MAPSS (telemetri + RUL). İndirme:

```bash
python 00_veri_indir.py --mimii-bilgi
python 00_veri_indir.py --cmapss-bilgi
```

Kaynak eşlemesi ve kayıt protokolü: [docs/veri-toplama-sartnamesi.md](docs/veri-toplama-sartnamesi.md)

## Dizin düzeni

```
00_*.py … 09_*.py     numaralı boru hattı adımları (çalıştırılabilir)
src/nabiz/            paylaşılan modüller (data, dsp, models, eval, edge, protocol, serving)
cpp/                  C++ ONNX Runtime çıkarım ikili dosyası (CMakeLists.txt, src/, testler/)
third_party/          vendored SDK — ör. onnxruntime/ (indirilir, git dışı)
tests/               birim + entegrasyon testleri
docs/                şartname, gereksinimler, test planı, yol haritası
data/                ham + işlenmiş veri (git dışı)
modeller/ deneyler/  eğitim çıktıları (git dışı) — modeller/edge/*.onnx dahil
sonuclar/            split özeti, parmak izi, saha test raporu (versiyonlu)
docker-compose.yml   mosquitto (MQTT) + toplayıcı — Faz 5 canlı doğrulama
```

## Test katmanları

Ayrıntılı plan: [docs/test-plani.md](docs/test-plani.md)

0. Veri/split doğrulama (sızıntı) · 1. DSP birim · 2. Model kalite (AUC/pAUC/ECE) ·
3. Domain shift / robüstluk · 4. Uç/donanım (gecikme/güç/nicemleme paritesi) ·
5. Protokol/entegrasyon (ağ arıza enjeksiyonu) · 6. CI regresyon kapıları ·
7. Saha kabul testi (yazılı teslimat)
