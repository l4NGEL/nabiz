"""Test katmanı 1 — titreşim analizi (rulman frekansları, order, bantlı zarf)."""
import numpy as np
import pytest

from nabiz.dsp import titresim as T


# --------------------------------------------------------------------------- #
# rulman arıza frekansları
# --------------------------------------------------------------------------- #
def test_rulman_frekanslari_6205_1772rpm():
    # CWRU tahrik ucu, ~1772 rpm — literatür değerleri
    f = T.rulman_ariza_frekanslari(1772)
    assert abs(f["fr"] - 29.53) < 0.1
    assert abs(f["BPFO"] - 105.9) < 1.0
    assert abs(f["BPFI"] - 159.9) < 1.0
    assert abs(f["BSF"] - 69.5) < 1.0
    # BPFO + BPFI = n * fr
    assert abs((f["BPFO"] + f["BPFI"]) - 9 * f["fr"]) < 1e-6


def test_rulman_bpfo_n_kat_ftf():
    f = T.rulman_ariza_frekanslari(1800, n_bilya=8)
    assert abs(f["BPFO"] - 8 * f["FTF"]) < 1e-9


def test_rulman_gecersiz_geometri():
    with pytest.raises(ValueError):
        T.rulman_ariza_frekanslari(1800, d_bilya_mm=50, d_pitch_mm=40)


# --------------------------------------------------------------------------- #
# bant geçiren
# --------------------------------------------------------------------------- #
def test_bant_gecir_bant_disi_bastirir():
    sr = 25600
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 5000 * t)
    y = T.bant_gecir(x, sr, 3000, 7000)
    # 500 Hz bileşeni büyük ölçüde gitmeli, 5 kHz kalmalı
    assert np.std(y) < np.std(x)
    from scipy.fft import rfft, rfftfreq
    frk = rfftfreq(y.size, 1 / sr)
    mag = np.abs(rfft(y))
    assert mag[np.argmin(np.abs(frk - 5000))] > 10 * mag[np.argmin(np.abs(frk - 500))]


def test_bant_gecir_gecersiz_bant():
    with pytest.raises(ValueError):
        T.bant_gecir(np.zeros(1000), 25600, 7000, 3000)


# --------------------------------------------------------------------------- #
# bantlı zarf spektrumu — rulman imzası
# --------------------------------------------------------------------------- #
def test_bantli_zarf_bpfo_tepesi():
    sr = 25600
    bpfo = 107.0
    tasiyici = 4500.0                       # ivmeölçer rezonansı
    n = 2 * sr
    # BPFO oranında darbe treni
    darbe = np.zeros(n)
    darbe[:: int(round(sr / bpfo))] = 1.0
    # sönümlü rezonans darbesi (impact response)
    k = np.arange(256)
    burst = np.sin(2 * np.pi * tasiyici * k / sr) * np.exp(-k / 30.0)
    x = np.convolve(darbe, burst, mode="same")
    x += 0.02 * np.random.default_rng(0).standard_normal(n)

    frekans, genlik = T.bantli_zarf_spektrumu(x, sr, bant=(3500, 5500))
    dusuk = (frekans > 20) & (frekans < 400)
    tepe = frekans[dusuk][np.argmax(genlik[dusuk])]
    assert abs(tepe - bpfo) < 3.0


# --------------------------------------------------------------------------- #
# order spektrumu
# --------------------------------------------------------------------------- #
def test_order_spektrumu_sabit_rpm_order3():
    sr, rpm = 20000, 1800.0
    fr = rpm / 60.0
    t = np.arange(2 * sr) / sr
    x = np.sin(2 * np.pi * 3 * fr * t) + 0.3 * np.sin(2 * np.pi * 1 * fr * t)
    order, genlik = T.order_spektrumu(x, sr, rpm, maks_order=8)
    tepe_order = order[np.argmax(genlik)]
    assert abs(tepe_order - 3.0) < 0.05


def test_computed_order_takibi_degisken_hiz():
    sr = 20000
    t = np.arange(3 * sr) / sr
    # hız 1500 -> 2100 rpm doğrusal sweep
    rpm_t = np.linspace(1500, 2100, t.size)
    fr_t = rpm_t / 60.0
    faz = 2 * np.pi * np.cumsum(fr_t) / sr
    x = np.sin(3 * faz)  # sabit 3. order bileşen
    order, genlik = T.hesaplanan_order_takibi(x, sr, rpm_t, maks_order=8)
    tepe_order = order[np.argmax(genlik)]
    assert abs(tepe_order - 3.0) < 0.1


def test_computed_order_takibi_kisa_sinyal_hata():
    with pytest.raises(ValueError):
        T.hesaplanan_order_takibi(np.zeros(100), 20000, 60.0)  # ~0.005 devir


# --------------------------------------------------------------------------- #
# spektral kurtosis bant seçimi
# --------------------------------------------------------------------------- #
def test_spektral_kurtosis_impulsif_bandi_bulur():
    sr = 25600
    rng = np.random.default_rng(0)
    t = np.arange(sr) / sr
    x = 0.05 * rng.standard_normal(t.size)
    # 5-6 kHz civarında impulsif enerji
    darbeler = np.zeros_like(t)
    darbeler[:: sr // 40] = 1.0
    x += np.convolve(darbeler, np.sin(2 * np.pi * 5500 * np.arange(100) / sr)
                     * np.exp(-np.arange(100) / 20), mode="same")
    alt, ust = T.spektral_kurtosis_bant(x, sr, n_bant=8)
    assert 0 <= alt < ust <= sr / 2
    assert alt < 5500 < ust or abs((alt + ust) / 2 - 5500) < sr / 8
