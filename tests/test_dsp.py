"""Test katmanı 1 — DSP önyüz birim testleri (bilinen sinyal, determinizm, kenar durum)."""
import numpy as np
import pytest
from scipy.fft import rfftfreq

from nabiz import config
from nabiz.dsp import features as F

SR = config.ORNEKLEME_HIZI


# --------------------------------------------------------------------------- #
# çerçeveleme
# --------------------------------------------------------------------------- #
def test_cerceve_sekli_ve_bindirme():
    x = np.arange(1000, dtype=float)
    c = F.cerceve(x, 100, 50)
    assert c.shape == (19, 100)
    np.testing.assert_array_equal(c[0, :50], x[:50])
    np.testing.assert_array_equal(c[1, :50], x[50:100])


def test_cerceve_kisa_sinyal_padlenir():
    c = F.cerceve(np.ones(10), 100, 50)
    assert c.shape == (1, 100)
    assert c[0, :10].sum() == 10.0
    assert c[0, 10:].sum() == 0.0


def test_cerceve_gecersiz_parametre():
    with pytest.raises(ValueError):
        F.cerceve(np.ones(100), 0, 10)


# --------------------------------------------------------------------------- #
# log-mel spektrogram
# --------------------------------------------------------------------------- #
def test_log_mel_sekli(sinus):
    lm = F.log_mel_spektrogram(sinus(1000))
    assert lm.shape[0] == config.N_MEL
    assert lm.shape[1] > 1
    assert np.all(np.isfinite(lm))


def test_log_mel_determinizm(sinus):
    x = sinus(750)
    np.testing.assert_array_equal(
        F.log_mel_spektrogram(x), F.log_mel_spektrogram(x)
    )


@pytest.mark.parametrize("f0", [200.0, 1000.0, 3500.0])
def test_sinus_dogru_mel_bandinda_tepe(sinus, f0):
    lm = F.log_mel_spektrogram(sinus(f0))
    tepe_bant = int(np.argmax(lm.mean(axis=1)))

    fb = F.mel_filtre_bankasi(SR, config.N_FFT, config.N_MEL, config.FMIN, config.FMAX)
    frekans = rfftfreq(config.N_FFT, 1.0 / SR)
    bin_idx = int(np.argmin(np.abs(frekans - f0)))
    beklenen_bant = int(np.argmax(fb[:, bin_idx]))

    assert abs(tepe_bant - beklenen_bant) <= 1


def test_beyaz_gurultu_kabaca_duz():
    rng = np.random.default_rng(0)
    lm = F.log_mel_spektrogram(rng.standard_normal(int(2.0 * SR)))
    orta_bantlar = lm.mean(axis=1)[5:-5]
    # sinüste bir bant ~10+ birim baskın olur; gürültüde orta bantlar yakın
    assert np.ptp(orta_bantlar) < 6.0


def test_sessizlik_sonlu_ve_negatif():
    lm = F.log_mel_spektrogram(np.zeros(int(1.0 * SR)))
    assert np.all(np.isfinite(lm))
    assert np.all(lm < 0)  # log(eps)


def test_nan_girdi_hata(sinus):
    x = sinus(500)
    x[100] = np.nan
    with pytest.raises(ValueError):
        F.log_mel_spektrogram(x)


def test_iki_boyutlu_girdi_hata():
    with pytest.raises(ValueError):
        F.log_mel_spektrogram(np.zeros((2, 1000)))


def test_fmax_nyquist_uzeri_hata():
    with pytest.raises(ValueError):
        F.mel_filtre_bankasi(SR, config.N_FFT, config.N_MEL, 50.0, SR)


def test_mel_filtre_bankasi_sekli():
    fb = F.mel_filtre_bankasi(SR, 1024, 64, 50.0, 8000.0)
    assert fb.shape == (64, 513)
    assert np.all(fb >= 0)
    assert np.all(fb.sum(axis=1) > 0)  # her filtrenin desteği var


# --------------------------------------------------------------------------- #
# zarf spektrumu (titreşim / rulman)
# --------------------------------------------------------------------------- #
def test_zarf_spektrumu_modulasyon_tepesi():
    t = np.arange(int(2.0 * SR)) / SR
    fc, fm = 2000.0, 120.0  # taşıyıcı 2 kHz, modülasyon 120 Hz (dış bilezik imzası benzeri)
    x = (1.0 + 0.6 * np.sin(2 * np.pi * fm * t)) * np.sin(2 * np.pi * fc * t)
    frekans, genlik = F.zarf_spektrumu(x, SR)
    dusuk = frekans < 500
    tepe_frekans = frekans[dusuk][np.argmax(genlik[dusuk])]
    assert abs(tepe_frekans - fm) < 5.0


def test_zarf_spektrumu_nan_hata():
    x = np.zeros(1000)
    x[0] = np.inf
    with pytest.raises(ValueError):
        F.zarf_spektrumu(x)


# --------------------------------------------------------------------------- #
# öznitelik vektörü
# --------------------------------------------------------------------------- #
def test_oznitelik_vektoru_uzunluk_ve_sonlu(sinus):
    v = F.oznitelik_vektoru(sinus(800))
    assert v.shape == (2 * config.N_MEL + 2,)
    assert np.all(np.isfinite(v))


def test_oznitelik_vektoru_rms_dogru(sinus):
    v = F.oznitelik_vektoru(sinus(500, genlik=1.0))
    rms = v[-2]
    assert abs(rms - 1 / np.sqrt(2)) < 0.02  # sinüs RMS = A/√2
