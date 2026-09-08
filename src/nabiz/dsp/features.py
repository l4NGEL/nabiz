"""Sinyal işleme önyüzü — çerçeveleme, log-mel spektrogram, zarf spektrumu.

Saf numpy/scipy: torch bağımlılığı yok, böylece birim testleri hafif ve hızlı.
İleride C++/CUDA uyarlaması (Faz 4) bu tanımların golden-vector'larına karşı
doğrulanır — bkz. tests/test_dsp.py ve docs/test-plani.md §1-2.

Mel ölçeği: HTK formülü (2595·log10(1+f/700)), üçgen filtreler alan-normalize
(Slaney tarzı enerji normalizasyonu). Bu seçim C++ tarafında birebir kopyalanır.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window, hilbert

from nabiz import config


def _sonlu_1b(x, ad: str = "sinyal") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{ad} 1B olmalı, shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{ad} boş")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{ad} sonlu değil (NaN/inf)")
    return x


def cerceve(x, cerceve_uzunluk: int, hop: int) -> np.ndarray:
    """1B sinyali (n_cerceve, cerceve_uzunluk) matrisine böl.

    Sinyal bir çerçeveden kısaysa sıfırla doldurulur → (1, cerceve_uzunluk).
    """
    x = _sonlu_1b(x)
    if cerceve_uzunluk <= 0 or hop <= 0:
        raise ValueError("cerceve_uzunluk ve hop > 0 olmalı")
    if len(x) < cerceve_uzunluk:
        x = np.pad(x, (0, cerceve_uzunluk - len(x)))
    n = 1 + (len(x) - cerceve_uzunluk) // hop
    idx = np.arange(cerceve_uzunluk)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filtre_bankasi(
    ornekleme_hizi: int, n_fft: int, n_mel: int, fmin: float, fmax: float
) -> np.ndarray:
    """(n_mel, n_fft//2 + 1) üçgen mel filtre bankası, alan-normalize."""
    nyquist = ornekleme_hizi / 2.0
    if fmax > nyquist:
        raise ValueError(f"fmax ({fmax}) > Nyquist ({nyquist})")
    if fmin < 0 or fmin >= fmax:
        raise ValueError(f"0 <= fmin < fmax gerekli (fmin={fmin}, fmax={fmax})")

    fft_frekans = rfftfreq(n_fft, 1.0 / ornekleme_hizi)
    mel_noktalar = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mel + 2)
    hz_noktalar = _mel_to_hz(mel_noktalar)

    fb = np.zeros((n_mel, fft_frekans.size), dtype=np.float64)
    for i in range(n_mel):
        alt, orta, ust = hz_noktalar[i], hz_noktalar[i + 1], hz_noktalar[i + 2]
        sol = (fft_frekans - alt) / (orta - alt)
        sag = (ust - fft_frekans) / (ust - orta)
        fb[i] = np.clip(np.minimum(sol, sag), 0.0, None)
        fb[i] *= 2.0 / (ust - alt)  # alan normalizasyonu (enerji korunumu)
    return fb


def log_mel_spektrogram(
    x,
    ornekleme_hizi: int | None = None,
    n_fft: int | None = None,
    hop: int | None = None,
    n_mel: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    pencere: str = "hann",
    eps: float = 1e-10,
) -> np.ndarray:
    """(n_mel, n_cerceve) log-mel spektrogram. Deterministik.

    Öntanımlar `nabiz.config` üzerinden gelir.
    """
    sr = ornekleme_hizi or config.ORNEKLEME_HIZI
    n_fft = n_fft or config.N_FFT
    hop = hop or config.HOP
    n_mel = n_mel or config.N_MEL
    fmin = config.FMIN if fmin is None else fmin
    fmax = config.FMAX if fmax is None else fmax

    x = _sonlu_1b(x)
    w = get_window(pencere, n_fft, fftbins=True)
    cerceveler = cerceve(x, n_fft, hop) * w
    guc = np.abs(rfft(cerceveler, axis=1)) ** 2          # (n_cerceve, n_fft//2+1)
    fb = mel_filtre_bankasi(sr, n_fft, n_mel, fmin, fmax)
    mel = guc @ fb.T                                      # (n_cerceve, n_mel)
    return np.log(mel.T + eps)                            # (n_mel, n_cerceve)


def zarf_spektrumu(
    x, ornekleme_hizi: int | None = None, n_fft: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert zarfının genlik spektrumu — rulman/titreşim arıza imzası için.

    Genlik-modülasyonlu sinyalde modülasyon frekansında tepe verir (dış/iç
    bilezik geçiş frekansı → arıza teşhisi).

    Döner: (frekans_hz, genlik).
    """
    sr = ornekleme_hizi or config.ORNEKLEME_HIZI
    x = _sonlu_1b(x)
    zarf = np.abs(hilbert(x))
    zarf = zarf - zarf.mean()
    n = n_fft or zarf.size
    genlik = np.abs(rfft(zarf, n=n))
    frekans = rfftfreq(n, 1.0 / sr)
    return frekans, genlik


def oznitelik_vektoru(x, ornekleme_hizi: int | None = None) -> np.ndarray:
    """Bir pencere için özet öznitelik — baseline modeller ve hızlı deneme.

    [log-mel bant ortalamaları | log-mel bant std | RMS | kurtosis]
    """
    x = _sonlu_1b(x)
    lm = log_mel_spektrogram(x, ornekleme_hizi)
    rms = float(np.sqrt(np.mean(x**2)))
    var = np.mean(x**2)
    kurt = float(np.mean(x**4) / (var**2 + 1e-12)) if var > 0 else 0.0
    return np.concatenate([lm.mean(axis=1), lm.std(axis=1), [rms, kurt]])
