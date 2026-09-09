"""Telemetri öznitelik çıkarımı — RPM, yük, sıcaklık, gerilim.

Düşük hızlı (1-10 Hz) kanallardan bir analiz penceresi için özet istatistik +
trend öznitelikleri. Eksik/NaN kanal graceful ele alınır (FR-01): kanal yoksa
o kanalın öznitelikleri 0.0.
"""
from __future__ import annotations

import numpy as np

from nabiz import config

_ISTATISTIKLER = ("ort", "std", "min", "maks", "aralik", "egim", "son", "delta")


def telemetri_oznitelik(
    seri: dict[str, np.ndarray],
    kanallar: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Kanal başına 8 öznitelik: ort/std/min/maks/aralik/egim/son/delta."""
    kanallar = kanallar or config.TELEMETRI_KANALLAR
    oz: dict[str, float] = {}
    for ad in kanallar:
        v = np.asarray(seri.get(ad, []), dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            for s in _ISTATISTIKLER:
                oz[f"t_{ad}_{s}"] = 0.0
            continue
        egim = float(np.polyfit(np.arange(v.size), v, 1)[0]) if v.size >= 2 else 0.0
        oz[f"t_{ad}_ort"] = float(v.mean())
        oz[f"t_{ad}_std"] = float(v.std())
        oz[f"t_{ad}_min"] = float(v.min())
        oz[f"t_{ad}_maks"] = float(v.max())
        oz[f"t_{ad}_aralik"] = float(np.ptp(v))
        oz[f"t_{ad}_egim"] = egim
        oz[f"t_{ad}_son"] = float(v[-1])
        oz[f"t_{ad}_delta"] = float(v[-1] - v[0])
    return oz


def telemetri_oznitelik_adlari(kanallar: tuple[str, ...] | None = None) -> list[str]:
    kanallar = kanallar or config.TELEMETRI_KANALLAR
    return [f"t_{ad}_{s}" for ad in kanallar for s in _ISTATISTIKLER]


def rejim_etiketi(
    rpm=None, yuk=None, rpm_nominal: float | None = None
) -> str:
    """RPM (+ opsiyonel yük) → ayrık çalışma rejimi etiketi."""
    if rpm is None:
        return "bilinmiyor"
    rpm_nominal = rpm_nominal or config.RPM_NOMINAL
    oran = float(np.mean(rpm)) / rpm_nominal
    hiz = (
        "dusuk" if oran < 0.6
        else "orta" if oran < 0.9
        else "anma" if oran <= 1.05
        else "asiri"
    )
    if yuk is None:
        return hiz
    y = float(np.mean(yuk))
    yb = "bos" if y < 0.25 else "yari" if y < 0.75 else "tam"
    return f"{hiz}/{yb}"


def sentetik_telemetri(
    sure_sn: float,
    rpm_nominal: float | None = None,
    hiz: float | None = None,
    seed: int = 0,
    anomali: bool = False,
) -> dict[str, np.ndarray]:
    """Gerçek telemetri yokken makul sahte kanallar (Faz 1 vekil).

    Anahtar `_zaman_sn` pencere-göreli zaman damgalarını taşır.
    """
    rpm_nominal = rpm_nominal or config.RPM_NOMINAL
    hiz = hiz or config.TELEMETRI_HIZI
    rng = np.random.default_rng(seed)
    n = max(2, int(round(sure_sn * hiz)))
    t = np.arange(n) / hiz

    rpm = rpm_nominal + rng.normal(0, 5, n) + 10 * np.sin(2 * np.pi * 0.05 * t)
    yuk = np.clip(0.6 + rng.normal(0, 0.03, n), 0.0, 1.0)
    sicaklik = 45.0 + (t / max(t[-1], 1e-9)) * (8.0 if anomali else 3.0) + rng.normal(0, 0.2, n)
    gerilim = 400.0 + rng.normal(0, 1.5, n)
    if anomali:
        rpm += rng.normal(0, 15, n)  # arıza → daha oynak devir

    return {
        "rpm": rpm,
        "yuk": yuk,
        "sicaklik": sicaklik,
        "gerilim": gerilim,
        "_zaman_sn": t,
    }
