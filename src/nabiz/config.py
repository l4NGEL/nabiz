"""Merkezi yol ve sabit tanımları.

Sinyal işleme öntanımları [docs/veri-toplama-sartnamesi.md] ile tutarlı tutulur.
Ortam değişkeni ile üzerine yazılabilir: NABIZ_VERI.
"""
from __future__ import annotations

import os
from pathlib import Path

# Depo kökü — bu dosya src/nabiz/config.py içinde
KOK = Path(__file__).resolve().parents[2]

VERI_KOK = Path(os.environ.get("NABIZ_VERI", KOK / "data"))
HAM_VERI = VERI_KOK / "ham"            # indirilen / üretilen ham veri
ISLENMIS_VERI = VERI_KOK / "islenmis"  # manifest, split, öznitelik
MODEL_DIZINI = KOK / "modeller"
DENEY_DIZINI = KOK / "deneyler"
SONUC_DIZINI = KOK / "sonuclar"


def dizinleri_hazirla() -> None:
    for d in (HAM_VERI, ISLENMIS_VERI, MODEL_DIZINI, DENEY_DIZINI, SONUC_DIZINI):
        d.mkdir(parents=True, exist_ok=True)


# --- Sinyal işleme öntanımları (akustik yol) ---
ORNEKLEME_HIZI = 16_000   # Hz — MIMII yerel; diğer akustik kaynaklar buna yeniden örneklenir
PENCERE_SN = 2.0          # analiz penceresi süresi
BINDIRME_SN = 1.0         # ardışık pencereler arası kayma
N_FFT = 1024
HOP = 512
N_MEL = 64
FMIN = 50.0
FMAX = 8_000.0            # <= Nyquist (ORNEKLEME_HIZI / 2)

# --- Titreşim / order analizi ---
TITRESIM_HIZI = 25_600           # Hz — şartname (min 3x ilgi bandı)
MAKS_ORDER = 10.0                # order spektrumu üst sınırı (shaft rate katı)
ORDER_COZUNURLUK = 0.01          # order grid adımı
ZARF_BANDI = (2_000.0, 6_000.0)  # demodülasyon bandı öntanımı (ivmeölçer rezonansı)
ORNEK_PER_DEVIR = 64             # hesaplanan order takibi çözünürlüğü

# SKF 6205-2RS (CWRU tahrik ucu rulmanı) geometrisi — order analizi öntanımı
RULMAN_6205 = {
    "n_bilya": 9,
    "d_bilya_mm": 7.94,
    "d_pitch_mm": 39.04,
    "temas_acisi_derece": 0.0,
}

# --- Telemetri ---
TELEMETRI_HIZI = 10.0            # Hz
TELEMETRI_KANALLAR = ("rpm", "yuk", "sicaklik", "gerilim")
RPM_NOMINAL = 1_800.0

# --- Split öntanımları ---
VARSAYILAN_SEED = 20_260_908
TEST_ORANI = 0.20
VAL_ORANI = 0.10
