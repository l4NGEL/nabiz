"""Veri manifesti oluşturma.

Manifest = her örneği (WAV kaydı) tanımlayan tablo. Sütunlar:

    dosya           veri kökÜNE GÖRE göreli yol, '/' ayraçlı  (portatif → parmak izi kararlı)
    veri_seti       'mimii' | 'cwru' | 'cmapss' | 'sentetik' | ...
    makine_tipi     'fan' | 'pump' | 'bearing' | ...
    makine_birimi   fiziksel ünite kimliği, tipe göre namespace'li ('fan/id_00')
    etiket          'normal' | 'anomali'
    ariza_tipi      'yok' | 'ic_bilezik' | 'dis_bilezik' | ...
    rejim           çalışma rejimi / koşul vekili (MIMII'de SNR: '-6dB' | '0dB' | '6dB')

Veri kökü `df.attrs['veri_kok']` içinde tutulur; mutlak yol için `mutlak_yollar(df)`.
Göreli yol sayesinde split parmak izi makineden bağımsızdır (CI kararlı).

Bölme (`splits.py`) bu tablo üzerinde çalışır; sızıntı kontrolleri `makine_birimi`
ve `dosya` sütunlarına dayanır.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_SURUM = 1

SUTUNLAR = [
    "dosya",
    "veri_seti",
    "makine_tipi",
    "makine_birimi",
    "etiket",
    "ariza_tipi",
    "rejim",
]

_MIMII_TIPLER = {"fan", "pump", "slider", "valve"}


def manifest_dogrula(df: pd.DataFrame) -> pd.DataFrame:
    """Şema ve temel sağlık kontrolleri. Sorun varsa ValueError."""
    eksik = set(SUTUNLAR) - set(df.columns)
    if eksik:
        raise ValueError(f"manifest sütunları eksik: {sorted(eksik)}")
    if df.empty:
        raise ValueError("manifest boş")
    gecersiz = set(df["etiket"].unique()) - {"normal", "anomali"}
    if gecersiz:
        raise ValueError(f"geçersiz etiket değerleri: {gecersiz}")
    if df["dosya"].duplicated().any():
        n = int(df["dosya"].duplicated().sum())
        raise ValueError(f"{n} duplike dosya yolu")
    karma = df.groupby("makine_birimi")["makine_tipi"].nunique()
    if (karma > 1).any():
        raise ValueError(f"birim birden çok tipe bağlı: {list(karma[karma > 1].index)}")
    return df


def mutlak_yollar(df: pd.DataFrame) -> list[Path]:
    """`dosya` (göreli) + `attrs['veri_kok']` → mutlak Path listesi."""
    kok = Path(df.attrs.get("veri_kok", "."))
    return [kok / d for d in df["dosya"]]


def _goreli(wav: Path, kok: Path) -> str:
    return wav.relative_to(kok).as_posix()


def _cerceve_kur(satirlar: list[dict], kok: Path, veri_seti: str) -> pd.DataFrame:
    df = manifest_dogrula(pd.DataFrame(satirlar, columns=SUTUNLAR))
    df.attrs["veri_kok"] = str(kok.resolve())
    df.attrs["veri_seti"] = veri_seti
    df.attrs["manifest_surum"] = MANIFEST_SURUM
    return df


def _dizinden_topla(kok: Path, veri_seti: str, tip_filtre: set[str] | None) -> list[dict]:
    satirlar: list[dict] = []
    for wav in sorted(kok.rglob("*.wav")):
        parcalar = wav.relative_to(kok).parts
        if len(parcalar) < 4:
            continue
        durum, birim, tip = parcalar[-2], parcalar[-3], parcalar[-4]
        if durum not in ("normal", "abnormal"):
            continue
        if tip_filtre is not None and tip not in tip_filtre:
            continue
        snr = parcalar[-5] if len(parcalar) >= 5 else "varsayilan"
        satirlar.append(
            {
                "dosya": _goreli(wav, kok),
                "veri_seti": veri_seti,
                "makine_tipi": tip,
                "makine_birimi": f"{tip}/{birim}",
                "etiket": "normal" if durum == "normal" else "anomali",
                "ariza_tipi": "yok" if durum == "normal" else "diger",
                "rejim": snr,
            }
        )
    return satirlar


def mimii_manifest(kok: str | Path) -> pd.DataFrame:
    """MIMII dizininden manifest.

    Beklenen düzen:  <kok>/<snr>/<tip>/id_<NN>/<normal|abnormal>/<dosya>.wav
    """
    kok = Path(kok)
    satirlar = _dizinden_topla(kok, "mimii", _MIMII_TIPLER)
    if not satirlar:
        raise FileNotFoundError(f"MIMII wav bulunamadı: {kok}")
    return _cerceve_kur(satirlar, kok, "mimii")


def dizin_manifest(kok: str | Path, veri_seti: str = "sentetik") -> pd.DataFrame:
    """MIMII benzeri dizin düzeninden genel manifest (sentetik veri dahil).

    Düzen:  <kok>/<tip>/id_<NN>/<normal|abnormal>/<dosya>.wav
    """
    kok = Path(kok)
    satirlar = _dizinden_topla(kok, veri_seti, None)
    if not satirlar:
        raise FileNotFoundError(f"wav bulunamadı: {kok}")
    return _cerceve_kur(satirlar, kok, veri_seti)


def sentetik_manifest(
    n_birim: int = 6,
    n_normal: int = 50,
    n_anomali: int = 10,
    seed: int = 0,
    tipler: tuple[str, ...] = ("fan", "pump"),
    rejimler: tuple[str, ...] = ("0dB", "6dB"),
) -> pd.DataFrame:
    """Test ve boru hattı denemesi için sahte manifest (gerçek dosya YOK)."""
    rng = np.random.default_rng(seed)
    satirlar: list[dict] = []
    for tip in tipler:
        for b in range(n_birim):
            birim = f"{tip}/id_{b:02d}"
            for etiket, adet in (("normal", n_normal), ("anomali", n_anomali)):
                durum = "normal" if etiket == "normal" else "abnormal"
                for i in range(adet):
                    satirlar.append(
                        {
                            "dosya": f"{birim}/{durum}/{i:04d}.wav",
                            "veri_seti": "sentetik",
                            "makine_tipi": tip,
                            "makine_birimi": birim,
                            "etiket": etiket,
                            "ariza_tipi": "yok" if etiket == "normal" else "diger",
                            "rejim": str(rng.choice(rejimler)),
                        }
                    )
    return _cerceve_kur(satirlar, Path("sanal"), "sentetik")
