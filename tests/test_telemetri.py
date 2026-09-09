"""Test katmanı 1 — telemetri öznitelik çıkarımı."""
import numpy as np
import pytest

from nabiz import config
from nabiz.dsp import telemetri as TL


def test_oznitelik_kanal_basina_sekiz():
    seri = {"rpm": np.array([1800.0, 1810, 1795, 1805])}
    oz = TL.telemetri_oznitelik(seri, kanallar=("rpm",))
    assert set(oz) == {
        "t_rpm_ort", "t_rpm_std", "t_rpm_min", "t_rpm_maks",
        "t_rpm_aralik", "t_rpm_egim", "t_rpm_son", "t_rpm_delta",
    }
    assert oz["t_rpm_ort"] == pytest.approx(1802.5)
    assert oz["t_rpm_son"] == 1805.0
    assert oz["t_rpm_delta"] == pytest.approx(5.0)
    assert oz["t_rpm_aralik"] == pytest.approx(15.0)


def test_oznitelik_egim_pozitif_trend():
    seri = {"sicaklik": np.linspace(40.0, 60.0, 20)}
    oz = TL.telemetri_oznitelik(seri, kanallar=("sicaklik",))
    assert oz["t_sicaklik_egim"] > 0
    assert oz["t_sicaklik_delta"] == pytest.approx(20.0)


def test_eksik_kanal_sifir_graceful():
    oz = TL.telemetri_oznitelik({"rpm": np.array([1800.0, 1801])}, kanallar=("rpm", "yuk"))
    assert all(oz[k] == 0.0 for k in oz if k.startswith("t_yuk_"))


def test_nan_ayiklanir():
    seri = {"rpm": np.array([1800.0, np.nan, 1802.0, np.inf])}
    oz = TL.telemetri_oznitelik(seri, kanallar=("rpm",))
    assert oz["t_rpm_ort"] == pytest.approx(1801.0)


def test_oznitelik_adlari_tutarli():
    adlar = TL.telemetri_oznitelik_adlari()
    tel = TL.sentetik_telemetri(2.0, seed=0)
    oz = TL.telemetri_oznitelik({k: v for k, v in tel.items() if k != "_zaman_sn"})
    assert set(adlar) == set(oz)
    assert len(adlar) == 8 * len(config.TELEMETRI_KANALLAR)


def test_rejim_etiketi_bantlar():
    assert TL.rejim_etiketi(rpm=900, rpm_nominal=1800) == "dusuk"
    assert TL.rejim_etiketi(rpm=1800, rpm_nominal=1800) == "anma"
    assert TL.rejim_etiketi(rpm=2200, rpm_nominal=1800) == "asiri"
    assert TL.rejim_etiketi(rpm=1800, yuk=0.1, rpm_nominal=1800) == "anma/bos"
    assert TL.rejim_etiketi(rpm=None) == "bilinmiyor"


def test_sentetik_telemetri_deterministik_ve_sekil():
    a = TL.sentetik_telemetri(3.0, seed=7)
    b = TL.sentetik_telemetri(3.0, seed=7)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a["_zaman_sn"].size == int(3.0 * config.TELEMETRI_HIZI)
    assert a["rpm"].size == a["_zaman_sn"].size


def test_sentetik_telemetri_anomali_daha_oynak():
    normal = TL.sentetik_telemetri(5.0, seed=1, anomali=False)
    ariza = TL.sentetik_telemetri(5.0, seed=1, anomali=True)
    assert ariza["rpm"].std() > normal["rpm"].std()
