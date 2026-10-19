"""Dosya düzeyinde alarm bütçesi (PR-01)."""
import numpy as np
import pytest

from nabiz.eval import metrics as MT
from nabiz.eval.alarm import (
    alarm_olaylari,
    butce_sureklilik,
    dosya_auc,
    dosya_sure_sn,
    dosyalara_indir,
    esik_fp_butcesi,
    isletme_dosya,
    isletme_sec,
    olay_sayisi,
)


def test_sure_ortusen_pencereler():
    # 0,1,2. pencereler: 2 sn, kayma 1 sn → 4 sn
    assert dosya_sure_sn(np.array([0, 1, 2]), pencere_sn=2.0, hop_sn=1.0) == pytest.approx(4.0)


def test_medyan_tek_sivri_pencereyi_yutar():
    skor = np.array([0.1, 0.1, 9.0, 4.0, 5.0, 6.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    dosya = np.array(["n", "n", "n", "a", "a", "a"])
    tip = np.array(["fan"] * 6)
    idx = np.array([0, 1, 2, 0, 1, 2])
    d = dosyalara_indir(skor, y, dosya, tip, idx, "medyan", 2.0, 1.0)
    yer = {k: i for i, k in enumerate(d["dosya"])}
    assert d["skor"][yer["n"]] == pytest.approx(0.1)
    assert d["skor"][yer["a"]] == pytest.approx(5.0)
    p90 = dosyalara_indir(skor, y, dosya, tip, idx, "p90", 2.0, 1.0)
    assert p90["skor"][yer["n"]] > 5.0


def test_kisa_kayit_esigi_normallerin_ustunde():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    esik = esik_fp_butcesi(y, s, toplam_saat=0.5, alarm_saat_basi=MT.PR01_ALARM_SAAT)
    assert esik > 3.0
    met = MT.esikte_metrikler(y, s, esik)
    assert met["fp"] == 0
    assert met["recall"] == 1.0


def _paket(skor, y, tip, sure):
    return {
        "skor": np.asarray(skor, dtype=np.float64),
        "y": np.asarray(y, dtype=int),
        "tip": np.asarray(tip),
        "sure_sn": np.asarray(sure, dtype=np.float64),
    }


def test_isletme_pr01_recall_sifir_yanlis_alarmla():
    val = _paket([1.0, 1.2, 1.1, 8.0, 9.0], [0, 0, 0, 1, 1], ["fan"] * 5, [10.0] * 5)
    te = _paket([1.05, 9.5, 0.4], [0, 1, 0], ["fan"] * 3, [10.0] * 3)
    r = isletme_dosya(val, te, MT.PR01_ALARM_SAAT)
    assert r["fp"] == 0
    assert r["recall"] == 1.0
    assert r["pr01"] is True


def test_olay_izole_sicak_sayilmaz():
    assert olay_sayisi(np.array([1, 0, 1, 0, 1]), k=2) == 0
    assert olay_sayisi(np.array([1, 1, 0, 1, 1, 1]), k=2) == 2


def test_sureklilik_izole_sivri_recall_birakir():
    normal = np.array([1.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0])
    anom = np.array([4.0, 4.0, 4.0, 4.0])
    skor = np.r_[normal, anom]
    y = np.r_[np.zeros(10), np.ones(4)].astype(int)
    dosya = np.array(
        [f"u/normal/{i:02d}.wav" for i in range(10)] + [f"u/abnormal/{i:02d}.wav" for i in range(4)]
    )
    paket = {
        "skor": skor,
        "y": y,
        "tip": np.array(["fan"] * 14),
        "sure_sn": np.full(14, 10.0),
        "dosya": dosya,
        "birim": np.array(["fan"] * 14),
        "oturum": np.array([d.rsplit("/", 1)[0] for d in dosya]),
    }
    r = butce_sureklilik(paket, paket, MT.PR01_ALARM_SAAT)
    assert r["k"] > 1
    assert r["recall"] == pytest.approx(1.0)
    assert r["olay_fp"] == 0
    assert r["pr01"] is True


def test_alarm_olayi_k_inci_dosyada_bir_kez_acilir():
    skor = np.array([0.0, 3.0, 3.0, 3.0, 0.0, 3.0])
    dosya = np.array([f"u/abnormal/{i:02d}.wav" for i in range(6)])
    paket = {
        "skor": skor,
        "y": np.ones(6, dtype=int),
        "tip": np.array(["fan"] * 6),
        "sure_sn": np.full(6, 10.0),
        "dosya": dosya,
        "birim": np.array(["fan/id_02"] * 6),
        "oturum": np.array(["u/abnormal"] * 6),
    }
    olay = alarm_olaylari(paket, esik=2.0, k=3)
    assert len(olay) == 1
    assert olay[0]["dosya"].endswith("03.wav")
    assert olay[0]["birim"] == "fan/id_02"
    assert alarm_olaylari(paket, esik=2.0, k=6) == []


def test_esit_recallde_en_uzun_sureklilik():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=24)
    anom = np.full(6, 40.0)
    dosya = [f"u/normal/{i:02d}.wav" for i in range(24)] + [
        f"u/abnormal/{i:02d}.wav" for i in range(6)
    ]
    paket = {
        "skor": np.r_[normal, anom],
        "y": np.r_[np.zeros(24), np.ones(6)].astype(int),
        "tip": np.array(["fan"] * 30),
        "sure_sn": np.full(30, 10.0),
        "dosya": np.array(dosya),
        "birim": np.array(["fan"] * 30),
        "oturum": np.array([d.rsplit("/", 1)[0] for d in dosya]),
    }
    sec = isletme_sec(paket, MT.PR01_ALARM_SAAT)
    assert sec["uygun"] is True
    assert sec["k"] == 6
    assert sec["val_recall"] == pytest.approx(1.0)


def test_dosya_auc_tip_ortalamasi():
    paket = _paket(
        [0.1, 0.2, 3.0, 4.0, 0.1, 5.0],
        [0, 0, 1, 1, 0, 1],
        ["fan", "fan", "fan", "fan", "pump", "pump"],
        [10.0] * 6,
    )
    rapor = dosya_auc(paket)
    assert rapor["tipler"]["fan"] == pytest.approx(1.0)
    assert rapor["makro_auc"] == pytest.approx(1.0)
