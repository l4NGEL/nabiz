"""Test katmanı 1 — mel_tensor sabit-boyut davranışı."""
import numpy as np
import pytest

from nabiz import config
from nabiz.dsp import features as F

SR = config.ORNEKLEME_HIZI


def _sinus(f0, sure_sn):
    t = np.arange(int(sure_sn * SR)) / SR
    return np.sin(2 * np.pi * f0 * t)


def test_mel_tensor_dogal_sekil():
    lm = F.mel_tensor(_sinus(500, config.PENCERE_SN))
    assert lm.shape[0] == config.N_MEL
    assert lm.shape[1] == F.varsayilan_cerceve_sayisi()


def test_mel_tensor_hedef_cerceve_kirpar():
    lm = F.mel_tensor(_sinus(500, 4.0), hedef_cerceve=40)
    assert lm.shape == (config.N_MEL, 40)


def test_mel_tensor_hedef_cerceve_doldurur():
    lm = F.mel_tensor(_sinus(500, 0.5), hedef_cerceve=61)
    assert lm.shape == (config.N_MEL, 61)
    assert np.all(np.isfinite(lm))


def test_mel_tensor_tam_esitse_dokunmaz():
    hedef = F.varsayilan_cerceve_sayisi()
    a = F.mel_tensor(_sinus(300, config.PENCERE_SN))
    b = F.mel_tensor(_sinus(300, config.PENCERE_SN), hedef_cerceve=hedef)
    np.testing.assert_array_equal(a, b)


def test_varsayilan_cerceve_sayisi_pozitif():
    assert F.varsayilan_cerceve_sayisi() > 1


def test_mel_tensor_nan_hata():
    x = _sinus(500, config.PENCERE_SN)
    x[10] = np.nan
    with pytest.raises(ValueError):
        F.mel_tensor(x)
