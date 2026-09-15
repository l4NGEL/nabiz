"""Test katmanı 2 — Platt kalibrasyonu."""
import numpy as np
import pytest

from nabiz.eval import metrics as MT
from nabiz.eval.kalibrasyon import PlattKalibratoru


@pytest.fixture
def skor_ve_etiket():
    rng = np.random.default_rng(0)
    s_normal = rng.normal(0, 1, 300)
    s_anomali = rng.normal(4, 1.5, 100)
    skor = np.r_[s_normal, s_anomali]
    y = np.r_[np.zeros(300), np.ones(100)].astype(int)
    idx = rng.permutation(len(skor))
    return skor[idx], y[idx]


def test_olasilik_araligi_ve_monotonluk(skor_ve_etiket):
    skor, y = skor_ve_etiket
    kal = PlattKalibratoru().egit(skor, y)
    p = kal.olasilik(skor)
    assert np.all((p >= 0) & (p <= 1))
    # yüksek skor -> yüksek olasılık (Platt tek değişkenli lojistik, monoton olmalı)
    sira = np.argsort(skor)
    assert np.all(np.diff(p[sira]) >= -1e-9)


def test_ece_kalibrasyon_sonrasi_iyilesir(skor_ve_etiket):
    skor, y = skor_ve_etiket
    kal = PlattKalibratoru().egit(skor, y)
    p = kal.olasilik(skor)

    # kalibre edilmemiş naif "olasılık": min-max normalize ham skor
    naif = (skor - skor.min()) / (skor.max() - skor.min())

    ece_kalibre = MT.ece(y, p)
    ece_naif = MT.ece(y, naif)
    assert ece_kalibre <= ece_naif + 1e-6


def test_tek_sinif_val_hata():
    with pytest.raises(ValueError, match="iki sınıf"):
        PlattKalibratoru().egit(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))


def test_uzunluk_uyusmazligi_hata():
    with pytest.raises(ValueError, match="uzunluğu farklı"):
        PlattKalibratoru().egit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))


def test_egitilmeden_olasilik_hata():
    with pytest.raises(RuntimeError):
        PlattKalibratoru().olasilik(np.array([0.1, 0.2]))


def test_yeni_skor_uzerinde_calisir(skor_ve_etiket):
    skor, y = skor_ve_etiket
    kal = PlattKalibratoru().egit(skor[:300], y[:300])
    p = kal.olasilik(skor[300:])
    assert p.shape == skor[300:].shape
