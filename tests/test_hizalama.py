"""Test katmanı 1 — çok-kanal hizalama (FR-01)."""
import numpy as np
import pytest

from nabiz.dsp import hizalama as H


def _ton(f0, sure_sn, sr, faz=0.0):
    t = np.arange(int(sure_sn * sr)) / sr
    return np.sin(2 * np.pi * f0 * t + faz)


def test_yeniden_ornekle_hedef_uzunluk():
    k = H.Kanal(_ton(50, 1.0, 8000), 8000, ad="akustik")
    y = H.yeniden_ornekle(k, 16000)
    assert y.ornekleme_hizi == 16000
    assert abs(y.veri.size - 16000) <= 2
    assert y.ad == "akustik"


def test_yeniden_ornekle_ayni_hiz_kopya():
    k = H.Kanal(_ton(50, 1.0, 8000), 8000)
    y = H.yeniden_ornekle(k, 8000)
    assert y.veri is not k.veri
    np.testing.assert_array_equal(y.veri, k.veri)


def test_hizala_ortak_hiz_ve_uzunluk():
    kanallar = {
        "akustik": H.Kanal(_ton(200, 2.0, 16000), 16000, ad="akustik"),
        "titresim": H.Kanal(_ton(200, 2.0, 25600), 25600, ad="titresim"),
    }
    s = H.hizala(kanallar, hedef_hiz=25600)
    assert set(s.kanallar) == {"akustik", "titresim"}
    n = s.ornek_sayisi
    assert all(v.size == n for v in s.kanallar.values())
    assert s.hizalama_hatasi_sn < 1e-3
    assert s.eksik_kanallar == []


def test_hizala_farkli_baslangic_ortak_aralik():
    kanallar = {
        "a": H.Kanal(_ton(100, 3.0, 10000), 10000, baslangic_sn=0.0, ad="a"),
        "b": H.Kanal(_ton(100, 3.0, 10000), 10000, baslangic_sn=0.5, ad="b"),
    }
    s = H.hizala(kanallar, hedef_hiz=10000)
    # ortak aralık [0.5, 3.0] -> 2.5 sn
    assert abs(s.ornek_sayisi - int(2.5 * 10000)) <= 2
    assert abs(s.ortak_baslangic_sn - 0.5) < 1e-6


def test_hizala_eksik_kanal_graceful():
    kanallar = {
        "akustik": H.Kanal(_ton(200, 1.0, 16000), 16000, ad="akustik"),
        "titresim": None,
    }
    s = H.hizala(kanallar, hedef_hiz=16000)
    assert s.eksik_kanallar == ["titresim"]
    assert list(s.kanallar) == ["akustik"]


def test_hizala_ortak_aralik_yoksa_hata():
    kanallar = {
        "a": H.Kanal(_ton(100, 1.0, 10000), 10000, baslangic_sn=0.0, ad="a"),
        "b": H.Kanal(_ton(100, 1.0, 10000), 10000, baslangic_sn=5.0, ad="b"),
    }
    with pytest.raises(H.HizalamaHatasi):
        H.hizala(kanallar, hedef_hiz=10000)


def test_hizala_tum_kanallar_eksikse_hata():
    with pytest.raises(H.HizalamaHatasi):
        H.hizala({"a": None, "b": None}, hedef_hiz=16000)


def test_capraz_korelasyon_bilinen_kayma():
    sr = 4000
    temiz = np.zeros(sr)
    temiz[1000:1010] = np.hanning(10)  # tek darbe
    a = temiz
    b = np.roll(temiz, 40)  # b, a'dan 40 örnek (10 ms) geç
    kayma = H.capraz_korelasyon_kaymasi(a, b, sr, maks_kayma_sn=0.05)
    assert abs(kayma - 0.010) < 1e-3


def test_telemetri_hizala_interpolasyon():
    kaynak_z = np.array([0.0, 1.0, 2.0])
    kaynak_d = np.array([10.0, 20.0, 30.0])
    hedef = np.array([0.5, 1.5])
    out = H.telemetri_hizala(kaynak_z, kaynak_d, hedef)
    np.testing.assert_allclose(out, [15.0, 25.0])


def test_telemetri_hizala_siralanmamis_kaynak():
    kaynak_z = np.array([2.0, 0.0, 1.0])
    kaynak_d = np.array([30.0, 10.0, 20.0])
    out = H.telemetri_hizala(kaynak_z, kaynak_d, np.array([0.5]))
    np.testing.assert_allclose(out, [15.0])


def test_telemetri_hizala_yetersiz_ornek_hata():
    with pytest.raises(ValueError):
        H.telemetri_hizala([0.0], [1.0], [0.0])


def test_kanal_nan_reddedilir():
    x = _ton(50, 1.0, 8000)
    x[10] = np.nan
    with pytest.raises(ValueError):
        H.Kanal(x, 8000)
