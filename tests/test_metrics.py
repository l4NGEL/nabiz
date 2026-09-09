"""Test katmanı 2 — metrikler (bilinen dizilerle)."""
import numpy as np
import pytest

from nabiz.eval import metrics as MT


def test_auc_mukemmel_ayrim():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert MT.auc(y, s) == 1.0


def test_auc_rastgele_yaklasik_yarim():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = rng.random(4000)
    assert abs(MT.auc(y, s) - 0.5) < 0.05


def test_auc_tek_sinif_hata():
    with pytest.raises(ValueError):
        MT.auc(np.zeros(10), np.arange(10))


def test_pauc_araligi_ve_mukemmel():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.0, 0.1, 0.2, 0.3, 0.9, 0.95, 0.97, 0.99])
    p = MT.pauc(y, s, maks_fpr=0.2)
    assert 0.5 <= p <= 1.0
    assert p == pytest.approx(1.0)


def test_esik_sec_f1_ayirir():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    s = np.r_[np.full(50, 0.2), np.full(50, 0.8)]
    esik = MT.esik_sec(y, s, hedef="f1")
    assert 0.2 < esik <= 0.8
    m = MT.esikte_metrikler(y, s, esik)
    assert m["f1"] == 1.0 and m["fpr"] == 0.0


def test_esik_sec_fpr_hedefli():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(500), np.ones(200)].astype(int)
    s = np.r_[rng.normal(0, 1, 500), rng.normal(3, 1, 200)]
    esik = MT.esik_sec(y, s, hedef="fpr", hedef_fpr=0.05)
    m = MT.esikte_metrikler(y, s, esik)
    assert abs(m["fpr"] - 0.05) < 0.04


def test_yanlis_alarm_saat():
    y = np.array([0, 0, 0, 0, 1])
    s = np.array([0.9, 0.9, 0.1, 0.1, 0.9])  # 2 normal pencere yanlış pozitif
    assert MT.yanlis_alarm_saat(y, s, esik=0.5, toplam_saat=4.0) == pytest.approx(0.5)


def test_ece_mukemmel_kalibrasyon_sifir():
    # olasılık = doğru sınıf oranı → iyi kalibre
    p = np.r_[np.full(100, 0.0), np.full(100, 1.0)]
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    assert MT.ece(y, p) == pytest.approx(0.0, abs=1e-9)


def test_ece_kotu_kalibrasyon_buyuk():
    p = np.full(200, 0.99)          # hep çok emin
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)  # ama yarısı yanlış
    assert MT.ece(y, p) > 0.4


def test_cmapss_gec_tahmin_daha_agir():
    gercek = np.array([50.0, 50.0])
    erken = np.array([40.0, 40.0])   # d = -10
    gec = np.array([60.0, 60.0])     # d = +10
    assert MT.cmapss_skoru(gercek, gec) > MT.cmapss_skoru(gercek, erken)
    assert MT.cmapss_skoru(gercek, gercek) == pytest.approx(0.0)


def test_bootstrap_ci_kapsar():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    s = np.r_[rng.normal(0, 1, 300), rng.normal(1.5, 1, 300)]
    ort, lo, hi = MT.bootstrap_ci(MT.auc, y, s, n=300, seed=0)
    assert lo < ort < hi
    assert 0.7 < ort < 0.95


def test_anomali_raporu_yapisi():
    rng = np.random.default_rng(3)
    y_te = np.r_[np.zeros(200), np.ones(80)].astype(int)
    s_te = np.r_[rng.normal(0, 1, 200), rng.normal(2, 1, 80)]
    y_va = np.r_[np.zeros(80), np.ones(30)].astype(int)
    s_va = np.r_[rng.normal(0, 1, 80), rng.normal(2, 1, 30)]
    tip = np.array(["fan"] * 140 + ["pump"] * 140)
    r = MT.anomali_raporu(y_te, s_te, y_va, s_va, kirilim={"makine_tipi": tip}, toplam_saat=2.0)
    assert {"auc", "pauc", "auc_alt", "auc_ust", "esikte", "kirilim", "yanlis_alarm_saat"} <= set(r)
    assert r["esik_kaynak"] == "val"
    assert set(r["kirilim"]["makine_tipi"]) <= {"fan", "pump"}
