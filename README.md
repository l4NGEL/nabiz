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
| 2 | Model aileleri (AE / CNN / CRNN / Transformer / füzyon) + değerlendirme | **sırada** |
| 3 | ONNX / INT8 / TensorRT + budama/distilasyon + uç benchmark | planlandı |
| 4 | C++ çıkarım yolu + CUDA mel çekirdeği | planlandı |
| 5 | Protobuf + MQTT/DDS + merkezi toplayıcı | planlandı |
| 6 | Saha testi simülasyonu + rapor + dashboard | planlandı |

Ayrıntı: [docs/yol-haritasi.md](docs/yol-haritasi.md)

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  Linux/mac: source .venv/bin/activate
pip install -e ".[dev]"
```

## Hızlı başlangıç (gerçek veri olmadan)

```bash
python 00_veri_indir.py --sentetik      # data/ham/sentetik/ altına küçük WAV seti (2 rejim)
python 01_split_dogrula.py               # manifest + 3 split + sızıntı kontrolü + parmak izi
python 02_oznitelik_cikar.py             # WAV → 2 sn pencere → 168 öznitelik → parquet
pytest -q                                # 68 birim testi
```

## Gerçek veri

MIMII (akustik endüstriyel makine — DCASE Task 2 tabanı), CWRU / Paderborn rulman
(titreşim), NASA C-MAPSS (telemetri + RUL). İndirme:

```bash
python 00_veri_indir.py --mimii-bilgi
```

Kaynak eşlemesi ve kayıt protokolü: [docs/veri-toplama-sartnamesi.md](docs/veri-toplama-sartnamesi.md)

## Dizin düzeni

```
00_*.py … 09_*.py     numaralı boru hattı adımları (çalıştırılabilir)
src/nabiz/            paylaşılan modüller (data, dsp, models, eval, edge, protocol, serving)
tests/               birim + entegrasyon testleri
docs/                şartname, gereksinimler, test planı, yol haritası
data/                ham + işlenmiş veri (git dışı)
modeller/ deneyler/  eğitim çıktıları (git dışı)
sonuclar/            split özeti, parmak izi, saha test raporu (versiyonlu)
```

## Test katmanları

Ayrıntılı plan: [docs/test-plani.md](docs/test-plani.md)

0. Veri/split doğrulama (sızıntı) · 1. DSP birim · 2. Model kalite (AUC/pAUC/ECE) ·
3. Domain shift / robüstluk · 4. Uç/donanım (gecikme/güç/nicemleme paritesi) ·
5. Protokol/entegrasyon (ağ arıza enjeksiyonu) · 6. CI regresyon kapıları ·
7. Saha kabul testi (yazılı teslimat)
