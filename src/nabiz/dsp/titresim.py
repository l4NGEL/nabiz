"""Titreşim analizi — rulman arıza frekansları, order spektrumu, bantlı zarf.

Order analizi: devre bağlı frekansları shaft-rate katına ('order') sabitler,
böylece değişken hızda arıza imzası sabit order'da kalır. `order_spektrumu`
sabit RPM için; `hesaplanan_order_takibi` (COT) değişken hız için açısal
yeniden örnekleme yapar.

Bantlı zarf (envelope) analizi: bir rezonans bandında demodülasyon → Hilbert
zarfı → FFT. Rulman darbeleri düşük frekanslı zarf modülasyonu üretir; tepe
BPFO/BPFI/BSF'de görülür.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, hilbert, sosfiltfilt, stft

from nabiz import config
from nabiz.dsp.dogrulama import pozitif, sonlu_1b


def rulman_ariza_frekanslari(
    rpm: float,
    n_bilya: int | None = None,
    d_bilya_mm: float | None = None,
    d_pitch_mm: float | None = None,
    temas_acisi_derece: float | None = None,
) -> dict[str, float]:
    """Rulman geometrisi + şaft hızından karakteristik arıza frekansları (Hz).

    FTF   kafes (fundamental train freq)
    BPFO  dış bilezik bilya geçiş frekansı
    BPFI  iç bilezik bilya geçiş frekansı
    BSF   bilya dönme frekansı
    """
    g = config.RULMAN_6205
    n = int(n_bilya if n_bilya is not None else g["n_bilya"])
    d = float(d_bilya_mm if d_bilya_mm is not None else g["d_bilya_mm"])
    D = float(d_pitch_mm if d_pitch_mm is not None else g["d_pitch_mm"])
    aci = float(temas_acisi_derece if temas_acisi_derece is not None else g["temas_acisi_derece"])
    pozitif(rpm, "rpm")
    if not (0 < d < D):
        raise ValueError(f"0 < d_bilya < d_pitch gerekli (d={d}, D={D})")

    fr = rpm / 60.0
    oran = (d / D) * np.cos(np.deg2rad(aci))
    ftf = 0.5 * fr * (1.0 - oran)
    bpfo = 0.5 * n * fr * (1.0 - oran)
    bpfi = 0.5 * n * fr * (1.0 + oran)
    bsf = 0.5 * (D / d) * fr * (1.0 - oran**2)
    return {"fr": fr, "FTF": ftf, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf}


def bant_gecir(x, ornekleme_hizi: float, alt_hz: float, ust_hz: float, order: int = 4) -> np.ndarray:
    """Sıfır-fazlı Butterworth bant geçiren (sosfiltfilt)."""
    x = sonlu_1b(x)
    sr = pozitif(ornekleme_hizi, "ornekleme_hizi")
    nyq = sr / 2.0
    if not (0 < alt_hz < ust_hz < nyq):
        raise ValueError(f"0 < alt < ust < Nyquist gerekli (alt={alt_hz}, ust={ust_hz}, nyq={nyq})")
    sos = butter(order, [alt_hz / nyq, ust_hz / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def bantli_zarf_spektrumu(
    x,
    ornekleme_hizi: float | None = None,
    bant: tuple[float, float] | None = None,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Demodülasyon bandında bant geçir → Hilbert zarfı → genlik spektrumu.

    Döner: (frekans_hz, genlik). Rulman arızasında tepe FTF/BPFO/BPFI/BSF'de.
    """
    sr = pozitif(ornekleme_hizi or config.TITRESIM_HIZI, "ornekleme_hizi")
    alt, ust = bant or config.ZARF_BANDI
    ust = min(ust, 0.98 * sr / 2.0)
    x = sonlu_1b(x)
    bantli = bant_gecir(x, sr, alt, ust)
    zarf = np.abs(hilbert(bantli))
    zarf = zarf - zarf.mean()
    n = n_fft or zarf.size
    genlik = np.abs(rfft(zarf, n=n))
    frekans = rfftfreq(n, 1.0 / sr)
    return frekans, genlik


def order_spektrumu(
    x,
    ornekleme_hizi: float,
    rpm: float,
    maks_order: float | None = None,
    order_cozunurluk: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sabit RPM order spektrumu — frekans eksenini shaft-rate'e böler.

    Döner: (order_grid, genlik).
    """
    x = sonlu_1b(x)
    sr = pozitif(ornekleme_hizi, "ornekleme_hizi")
    fr = pozitif(rpm, "rpm") / 60.0
    maks_order = maks_order or config.MAKS_ORDER
    d_order = order_cozunurluk or config.ORDER_COZUNURLUK

    spek = np.abs(rfft(x - x.mean()))
    frekans = rfftfreq(x.size, 1.0 / sr)
    orderlar = frekans / fr

    grid = np.arange(0.0, maks_order + d_order, d_order)
    genlik = np.interp(grid, orderlar, spek, left=0.0, right=0.0)
    return grid, genlik


def hesaplanan_order_takibi(
    x,
    ornekleme_hizi: float,
    rpm_profili,
    ornek_per_devir: int | None = None,
    maks_order: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Değişken hız için açısal yeniden örnekleme (COT) → order spektrumu.

    rpm_profili: skaler VEYA sinyal boyunca örneklenmiş hız dizisi.
    Döner: (order, genlik). Hızdan bağımsız arıza imzası.
    """
    x = sonlu_1b(x)
    sr = pozitif(ornekleme_hizi, "ornekleme_hizi")
    opd = int(ornek_per_devir or config.ORNEK_PER_DEVIR)

    if np.isscalar(rpm_profili):
        fr = np.full(x.size, pozitif(float(rpm_profili), "rpm") / 60.0)
    else:
        rp = sonlu_1b(rpm_profili, "rpm_profili")
        if np.any(rp <= 0):
            raise ValueError("rpm_profili tüm değerleri > 0 olmalı")
        tp = np.linspace(0.0, x.size / sr, rp.size)
        fr = np.interp(np.arange(x.size) / sr, tp, rp) / 60.0

    theta = 2.0 * np.pi * np.cumsum(fr) / sr          # kümülatif açı (rad)
    toplam_devir = float(theta[-1] / (2.0 * np.pi))
    n_ornek = int(toplam_devir * opd)
    if n_ornek < 16:
        raise ValueError(f"COT için sinyal çok kısa (≈{toplam_devir:.2f} devir)")

    hedef_theta = np.arange(n_ornek) * (2.0 * np.pi / opd)
    x_aci = np.interp(hedef_theta, theta, x)

    spek = np.abs(rfft(x_aci - x_aci.mean()))
    orderlar = rfftfreq(n_ornek, d=1.0 / opd)          # devir başına çevrim = order
    if maks_order:
        m = orderlar <= maks_order
        return orderlar[m], spek[m]
    return orderlar, spek


def spektral_kurtosis_bant(
    x,
    ornekleme_hizi: float,
    n_bant: int = 16,
    nperseg: int | None = None,
) -> tuple[float, float]:
    """En impulsif frekans bandını spektral kurtosis ile seç (basit kurtogram).

    Döner: (alt_hz, ust_hz) — `bantli_zarf_spektrumu` için demodülasyon bandı.
    """
    x = sonlu_1b(x)
    sr = pozitif(ornekleme_hizi, "ornekleme_hizi")
    nperseg = nperseg or max(256, int(2 ** np.ceil(np.log2(x.size / 64))))
    f, _t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    mag2 = np.abs(Z) ** 2
    m2 = mag2.mean(axis=1)
    m4 = (mag2 ** 2).mean(axis=1)
    sk = m4 / (m2 ** 2 + 1e-20) - 2.0                  # frekans binine göre SK

    bant_bin = max(1, sk.size // n_bant)
    blok_sk = [sk[i : i + bant_bin].mean() for i in range(0, sk.size - bant_bin + 1, bant_bin)]
    en_iyi = int(np.argmax(blok_sk))
    alt = f[en_iyi * bant_bin]
    ust = f[min(sk.size - 1, (en_iyi + 1) * bant_bin)]
    if ust <= alt:
        ust = min(sr / 2.0 * 0.98, alt + sr / (2 * n_bant))
    return float(alt), float(ust)
