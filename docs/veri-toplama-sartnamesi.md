# Veri Toplama Şartnamesi

**Belge no:** NABIZ-DVS-001 · **Sürüm:** 0.1 (taslak) · **Tarih:** 2026-09-08

İlgili gereksinimler: FR-01, RB-01, RB-02, QR-01
İlgili kod: [`src/nabiz/data/manifest.py`](../src/nabiz/data/manifest.py), [`src/nabiz/data/splits.py`](../src/nabiz/data/splits.py)

---

## 1. Amaç ve kapsam

Dönen/tekrarlı çalışan teçhizatın (fan, pompa, rulman, motor, valf) **normal ve
arızalı** durumlarını ayırt etmek için çok-sensörlü veri toplama tanımı. Bu belge:
model geliştirme öncesi hangi verinin, hangi koşulda, hangi etiketle toplanacağını
sabitler; kamuya açık veri setlerini bu tanıma eşler; bölme (train/val/test)
kurallarını ve izlenebilirliği tanımlar.

Kapsam dışı: görüntü/termal sensörler (ayrı şartname), sistem seviyesi kabul testi
(bkz. [test-plani.md](test-plani.md) §7).

## 2. Sensör tanımı

| Kanal | Sensör | Örnekleme | Çözünürlük | Montaj / not |
|-------|--------|-----------|------------|--------------|
| Akustik | Elektret/MEMS mikrofon | 16 kHz | 16-bit | Teçhizattan 10–50 cm, rüzgâr koruması; arka plan gürültüsü kaydedilir |
| Titreşim | IEPE ivmeölçer (±50 g) | 25.6 kHz (min 3× ilgi bandı) | 24-bit | Rulman yatağına radyal + eksenel; manyetik taban değil, cıvatalı |
| Telemetri | PLC/SCADA çıkışı | 1–10 Hz | — | RPM, yük/akım, giriş-çıkış sıcaklığı, besleme gerilimi |

Akustik ve titreşim **eş zamanlı** kaydedilir (ortak tetik veya PTP/GPS zaman
damgası, kayma < 1 ms). Telemetri en yakın zaman damgasıyla hizalanır.

## 3. Çalışma rejimleri

Model her rejimi görmeli **veya** rejim bilinçli olarak test'e ayrılmalı
(domain-shift değerlendirmesi, RB-01). Kaydedilecek rejim eksenleri:

- **Devir:** anma devrinin %50 / %75 / %100 / %110'u
- **Yük:** boş / yarı / tam / aşırı
- **Sıcaklık:** soğuk başlangıç / kararlı rejim
- **Arka plan:** sessiz / atölye gürültüsü / komşu makine çalışıyor
- **Kurulum varyasyonu:** sensör yeniden konumlandırma, kablo yönü (parazit için)

Her (teçhizat birimi × rejim × durum) hücresi için **en az 60 sn** kesintisiz kayıt;
pencere başına 2 sn, 1 sn bindirme → hücre başına ≥ 59 pencere.

## 4. Kayıt protokolü

1. Sensör kalibrasyonu (ivmeölçer hassasiyeti mV/g, mikrofon referans tonu 94 dB SPL).
2. 5 sn ortam kaydı (makine kapalı) — gürültü tabanı.
3. Makineyi hedef rejime getir, 30 sn oturma süresi.
4. 60–300 sn kesintisiz kayıt; kesme yok, kazanç sabit (AGC kapalı).
5. Meta veri formu: birim seri no, rejim, ortam, tarih/saat, operatör, arıza
   enjekte edildiyse tip + şiddet + yöntem.
6. Ham dosya adı: `<birim>/<rejim>/<durum>/<UTC-zaman-damgası>.wav` + eş `.json` meta.

## 5. Etiket şeması

| Alan | Değerler | Not |
|------|----------|-----|
| `etiket` | `normal`, `anomali` | Birincil ikili hedef (gözetimsiz AE bunu kullanır) |
| `ariza_tipi` | `yok`, `ic_bilezik`, `dis_bilezik`, `bilya`, `dengesizlik`, `gevseklik`, `kavitasyon`, `diger` | Gözetimli sınıflandırma dalı |
| `siddet` | `yok`, `baslangic`, `orta`, `ileri` | Ordinal |
| `rul_ref` | tam sayı (çevrim) veya `null` | Yalnız run-to-failure kayıtlarında |
| `kaynak` | `saha`, `enjekte`, `kamuya_acik` | Enjekte = kontrollü kusurlu parça |

Etiketleme kuralı: arıza tipi ve şiddeti **fiziksel muayene / enjekte kusur
kaydından** gelir, dinleyerek değil. Belirsiz kayıt `diger` + serbest not.

## 6. Veri kalitesi kabul kriterleri

Bir kayıt aşağıdakileri sağlamazsa **elenir** (`01_split_dogrula.py` raporlar):

- Kliplenme oranı < %0.1 (|x| ≥ 0.99 örnek yüzdesi)
- Sessiz segment (RMS < gürültü tabanı + 3 dB) toplam < %5
- Örnekleme hızı ve kanal sayısı beklenenle aynı
- Süre ≥ 55 sn (hücre başına ≥ 55 pencere)
- Meta `.json` tam ve şema geçerli
- Sınıf başına min: normal ≥ 200 pencere/birim, anomali ≥ 40 pencere/birim

## 7. Bölme stratejisi

Üç bölme, üçü de üretilir ve raporlanır ([`splits.py`](../src/nabiz/data/splits.py)):

| Strateji | Kural | Amaç | Sızıntı kontrolü |
|----------|-------|------|------------------|
| `birim-ici` | Her birim train/val/test'e katkı verir; bir **dosya** asla iki bölmede olmaz. Anomali örnekleri yalnız test'te. | Standart MIMII değerlendirmesi | dosya düzeyi |
| `birim-disi` | Bütün birimler tek bölmede; makine tipine göre tabakalı. Bir **birim** asla iki bölmede olmaz. | Görülmemiş fiziksel üniteye genelleme | birim (`makine_birimi`) düzeyi |
| `rejim-holdout` | Seçili çalışma rejim(ler)i = hedef domain (test); kalan = kaynak (train/val). | Domain-shift (RB-01) | dosya düzeyi + rejim ayrımı |

Sızıntı kontrolü **CI kapısıdır** — ihlalde build başarısız
([test-plani.md](test-plani.md) §0, `tests/test_splits.py`).

## 8. Kamuya açık kaynak eşlemesi

| Set | Sağladığı | Örnekleme | Lisans | Şartname karşılığı |
|-----|-----------|-----------|--------|--------------------|
| **MIMII** (Zenodo 3384388) | fan/pump/slider/valve, normal+anomali, 4 birim/tip, 3 SNR | 16 kHz | CC BY 4.0 | Akustik; rejim = SNR koşulu; birim = `id_00..06` |
| **MIMII DUE** (DCASE'20 T2) | Kaynak/hedef domain ayrımı hazır | 16 kHz | CC BY 4.0 | Domain-shift (RB-01) referans kümesi |
| **CWRU Bearing** | İç/dış bilezik + bilya arızası, 4 yük | 12 / 48 kHz | Akademik kullanım | Titreşim; `ariza_tipi` + yük rejimi |
| **Paderborn (PU)** | 32 rulman, gerçek + enjekte hasar | 64 kHz | CC BY 4.0 | Titreşim çapraz-veri (CWRU'da eğit → PU'da test) |
| **IMS Bearing (NASA)** | Run-to-failure, 4 rulman | 20 kHz | Kamu malı | `rul_ref`, zamansal split |
| **NASA C-MAPSS / N-CMAPSS** | Turbofan telemetri + RUL | çevrim bazlı | Kamu malı | Telemetri dalı, C-MAPSS asimetrik skor |

Yeniden örnekleme: tüm akustik kaynaklar 16 kHz'e; titreşim kendi bandında tutulur,
öznitelik çıkarımı örnekleme hızına göre parametrelenir.

## 9. Sürüm ve izlenebilirlik (QR-01)

- Ham veri **DVC** ile izlenir (`data/` git dışı; `.dvc` işaretçileri commit edilir).
- `01_split_dogrula.py` her bölme için **SHA-256 parmak izi** üretir
  (`sonuclar/split-parmak-izleri.json`) — sıralı `(dosya, bölme)` çiftleri üzerinden.
- Saha test raporu ve eğitim koşuları bu parmak izini alıntılar; CI parmak izinin
  değişmediğini doğrular (kasıtlı güncelleme dışında).
- Manifest şeması sürümlüdür (`manifest.py: MANIFEST_SURUM`).

## 10. Açık maddeler

- [ ] Telemetri kanalları için gerçek saha kaynağı (şu an C-MAPSS vekil)
- [ ] Titreşim + akustik eş zamanlı gerçek kayıt (kamuya açık setlerde ayrı)
- [ ] Enjekte arıza prosedürünün etik/güvenlik onayı (gerçek teçhizatta)
