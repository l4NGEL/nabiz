# Faz 2 — model karşılaştırması

- veri: **GERÇEK MIMII** (Zenodo 3384388) — fan + pump, 6dB + 0dB, 19.510 dosya, 8 fiziksel ünite (4 fan + 4 pump), 175.590 pencere
- öznitelik yolu: a_ + z_ + t_ (168d)  ·  mel yolu: log-mel tensör (aktif)
- torch: var  ·  tam mod (epok=120/100). birim_disi hem hızlı (epok=8) hem tam modda ayrı çalıştırıldı — sayılar epoka duyarsız/kararlı (bkz. alt not).
- gözetimsiz (yalnız normal eğitim); eşik validasyondan F1 ile
- **Not:** 2026-09-08'de başlayan sentetik-veri geliştirme evresinin ardından gerçek MIMII ile koşulan NİHAİ sonuç. Yol boyunca **kritik bir veri sızıntısı bulundu ve düzeltildi** (bkz. altta) — önceki "AUC≈1" iddiaları geçersizdi, bu tablo düzeltme SONRASI temiz özniteliklerle üretildi.

## Sonuç tablosu

| strateji | model | AUC | AUC_CI | pAUC | F1 | precision | recall | FPR | yanlış_alarm/sa |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| birim_disi | rms_esigi | 0.473 | [0.465, 0.481] | 0.522 | 0.188 | 0.179 | 0.198 | 0.204 | 367.3 |
| birim_disi | izolasyon_ormani | 0.438 | [0.432, 0.445] | 0.500 | 0.301 | 0.179 | 0.957 | 0.988 | 1778.5 |
| birim_disi | mahalanobis | 0.398 | [0.392, 0.405] | 0.501 | 0.218 | 0.144 | 0.449 | 0.598 | 1075.8 |
| birim_disi | dense_ae | 0.346 | [0.340, 0.353] | 0.484 | 0.300 | 0.178 | 0.943 | 0.978 | 1760.0 |
| birim_disi | conv_ae | 0.430 | [0.423, 0.435] | 0.481 | 0.310 | 0.184 | 1.000 | 0.999 | 1798.2 |
| birim_disi | crnn_ae | 0.437 | [0.429, 0.442] | 0.484 | 0.308 | 0.183 | 0.991 | 0.997 | 1793.8 |
| birim_disi | spek_transformer_ae | 0.511 | [0.503, 0.518] | 0.486 | 0.297 | 0.176 | 0.947 | 0.996 | 1793.5 |
| birim_disi | fusion_ae | 0.399 | [0.393, 0.405] | 0.477 | 0.310 | 0.183 | 1.000 | 1.000 | 1800.0 |
| rejim_holdout | rms_esigi | 0.587 | [0.581, 0.592] | 0.617 | 0.361 | 0.877 | 0.228 | 0.008 | 14.2 |
| rejim_holdout | izolasyon_ormani | 0.455 | [0.451, 0.460] | 0.502 | 0.331 | 0.201 | 0.943 | 0.927 | 1668.8 |
| rejim_holdout | mahalanobis | **0.661** | [0.657, 0.665] | 0.600 | 0.341 | 0.206 | 0.998 | 0.951 | 1711.7 |
| rejim_holdout | dense_ae | 0.637 | [0.633, 0.642] | 0.577 | 0.364 | 0.246 | 0.695 | 0.525 | 945.2 |
| rejim_holdout | conv_ae | 0.575 | [0.571, 0.579] | 0.491 | 0.379 | 0.237 | 0.947 | 0.754 | 1357.5 |
| rejim_holdout | crnn_ae | 0.474 | [0.470, 0.478] | 0.486 | 0.357 | 0.223 | 0.897 | 0.771 | 1387.7 |
| rejim_holdout | spek_transformer_ae | 0.586 | [0.582, 0.591] | 0.532 | 0.338 | 0.203 | 0.996 | 0.964 | 1735.3 |
| rejim_holdout | fusion_ae | 0.563 | [0.559, 0.568] | 0.511 | 0.351 | 0.215 | 0.950 | 0.854 | 1536.9 |

## Makine tipine göre AUC kırılımı

**birim_disi** (görülmemiş ünite — genelde zayıf):
fan/pump → rms_esigi 0.400/0.654 · izolasyon_ormani 0.568/0.503 · mahalanobis 0.493/0.541 · dense_ae 0.402/0.515 · conv_ae 0.491/0.522 · crnn_ae 0.507/0.477 · spek_transformer_ae 0.621/0.432 · fusion_ae 0.491/0.425

**rejim_holdout** (görülmemiş SNR — tip içinde ŞAŞIRTICI DERECEDE İYİ):
fan/pump → rms_esigi 0.489/0.664 · izolasyon_ormani 0.601/0.478 · **mahalanobis 0.803/0.861** · **dense_ae 0.715/0.841** · conv_ae 0.751/0.554 · crnn_ae 0.562/0.471 · spek_transformer_ae 0.774/0.653 · fusion_ae 0.679/0.670

**Önemli:** rejim_holdout'un TİP-İÇİ (fan-only veya pump-only) AUC'si, birleşik (fan+pump havuzlanmış) AUC'den çok daha yüksek (örn. mahalanobis birleşik 0.661 vs tip-içi 0.803/0.861). Bu, iki farklı makine tipinin skor dağılımlarının farklı ölçeklerde olmasından kaynaklanıyor — havuzlanmış AUC bu farkı "gürültü" gibi görüyor. **Tek makine tipi için gerçek SNR-domain-shift performansı 0.80-0.86 AUC — literatürle uyumlu, iyi bir sonuç.**

## Domain-shift (RB-01): birim_disi vs rejim_holdout (birleşik)

| model | birim_disi AUC | rejim_holdout AUC | Δ |
|---|---:|---:|---:|
| rms_esigi | 0.473 | 0.587 | +0.114 |
| izolasyon_ormani | 0.438 | 0.455 | +0.017 |
| mahalanobis | 0.398 | 0.661 | **+0.263** |
| dense_ae | 0.346 | 0.637 | **+0.291** |
| conv_ae | 0.430 | 0.575 | +0.145 |
| crnn_ae | 0.437 | 0.474 | +0.037 |
| spek_transformer_ae | 0.511 | 0.586 | +0.075 |
| fusion_ae | 0.399 | 0.563 | +0.164 |

**Bulgu: `rejim_holdout` (görülmemiş SNR koşulu) sistematik olarak `birim_disi`den (görülmemiş fiziksel ünite) daha kolay** — her model için pozitif Δ, en belirgini dense_ae/mahalanobis'te (+0.26-0.29 puan). Yorum: SNR kayması global/homojen bir gürültü değişimi; ünite-ünite mekanik farklılık (montaj, aşınma, üretim toleransı) çok daha idiosinkratik bir dağılım kayması yaratıyor. Bu, DCASE/MIMII literatüründeki "unseen machine ID" alt-görevinin bilinen zorluğuyla tutarlı.

**`birim_disi`'de bazı modeller rastgeleden KÖTÜ** (dense_ae 0.346, mahalanobis 0.398) — bu epok sayısına duyarsız (mahalanobis/izolasyon_ormani tek-seferlik fit, epok kavramı yok; dense_ae hızlı [epok=8, AUC 0.337] ile tam [epok=120, AUC 0.346] modda neredeyse aynı) — **gerçek bir cross-machine genelleme zorluğu, eğitim eksikliği değil.** Görülmemiş ünitenin "normal" akustik imzası, eğitim ünitelerinden öğrenilen normal prototipinden, o ünitenin GERÇEK arızalarından daha uzak duruyor.

## Kalibrasyon + OOD (dense_ae)

| strateji | ECE ham | ECE kalibre | OOD AUC (gürültü) | OOD AUC (karışık) |
|---|---:|---:|---:|---:|
| birim_disi | 0.145 | 0.085 | 0.950 | 0.949 |
| rejim_holdout | 0.193 | 0.039 | 0.999 | 0.999 |

OOD ayrımı her iki stratejide de mükemmel (~0.95-1.0) — model gürültü/yapısı-bozulmuş girdiyi güvenle "anomali" işaretliyor, "kendinden emin yanlış" riski düşük. Kalibrasyon rejim_holdout'ta belirgin iyileşiyor (0.193→0.039); birim_disi'de daha mütevazı (0.145→0.085).

## Düzeltilen kritik bulgu: sentetik telemetri etiket sızıntısı

Gerçek pump verisiyle ilk (düzeltme öncesi) koşuda `fusion_ae` AUC **1.000** vermişti. Kök neden: `sentetik_telemetri(anomali=...)` gerçek etikete göre RPM/sıcaklık gürültüsünü deterministik artırıyordu — telemetri dalı cevabı doğrudan kodluyordu, gerçek yetenek değildi. **Düzeltildi** (`anomali` parametresi kaldırıldı, regresyon testi eklendi — commit 775e6ad).

## Genel değerlendirme

Bu sayılar sentetik verideki "AUC≈1" iddialarından çok daha mütevazı — ama **gerçek, savunulabilir ve MIMII/DCASE literatürüyle tutarlı**. En iyi tekil sonuç: tip-içi `mahalanobis`/`dense_ae` rejim_holdout AUC 0.80-0.86. En önemli mühendislik bulgusu: cross-unit generalization (birim_disi) gerçek MIMII'de literatürle uyumlu şekilde çok zor, ve bu zorluk ampirik olarak (epok-bağımsız, iki ayrı koşuda) doğrulandı.
