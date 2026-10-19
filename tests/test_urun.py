"""Ürün noktası: ünite uyarlaması, tip başına model, işletme bütçesi."""
import numpy as np
import pytest

from nabiz.data.uyarlama import BirimOlcek
from nabiz.eval.urun import degerlendir_diziler
from nabiz.models.tip_dedektor import TipDedektoru


def _kaydirilmis_unite():
    """Görülmemiş ünitenin normali, arızadan daha uzak (filo merkezine göre)."""
    rng = np.random.default_rng(0)
    n = 240
    Xtr = rng.normal(0.0, 0.35, (n, 4))
    ytr = np.zeros(n, dtype=int)
    tiptr = np.array(["fan"] * n)
    birtr = np.array(["fan/id_00"] * n)

    def parti(n_normal, n_anomali, birim):
        X = np.vstack(
            [
                rng.normal(6.0, 0.35, (n_normal, 4)),
                rng.normal(2.0, 0.35, (n_anomali, 4)),
            ]
        )
        y = np.r_[np.zeros(n_normal), np.ones(n_anomali)].astype(int)
        tip = np.array(["fan"] * (n_normal + n_anomali))
        bir = np.array([birim] * (n_normal + n_anomali))
        return X, y, tip, bir

    Xva, yva, tipva, birva = parti(80, 40, "fan/id_98")
    Xte, yte, tipte, birte = parti(200, 80, "fan/id_99")
    return (
        Xtr, ytr, tiptr, birtr,
        Xva, yva, tipva, birva,
        Xte, yte, tipte, birte,
    )


def test_uyarlama_gorulmemis_unitenin_merkezini_sifirlar():
    rng = np.random.default_rng(1)
    Xtr = rng.normal(0.0, 1.0, (40, 3))
    ytr = np.zeros(40, dtype=int)
    birtr = np.array(["u0"] * 40)
    olcek = BirimOlcek().egit(Xtr, birtr, ytr)
    X = rng.normal(5.0, 0.2, (30, 3))
    out = olcek.uygula(X, np.array(["u1"] * 30), yeniden=True)
    assert np.abs(np.median(out, axis=0)).max() < 0.15


def test_uyarlama_egitim_istatistigini_korur():
    rng = np.random.default_rng(2)
    Xtr = np.vstack([rng.normal(0, 1, (40, 2)), rng.normal(10, 1, (40, 2))])
    ytr = np.zeros(80, dtype=int)
    birtr = np.array(["a"] * 40 + ["b"] * 40)
    olcek = BirimOlcek().egit(Xtr, birtr, ytr)
    X = np.full((20, 2), 10.0)
    out = olcek.uygula(X, np.array(["b"] * 20), yeniden=False)
    assert np.abs(out).mean() < 1.5


def test_tip_dedektor_iki_olcegi_ayirir():
    rng = np.random.default_rng(3)
    fan_n = rng.normal(0.0, 0.4, (80, 2))
    fan_a = rng.normal(3.0, 0.4, (40, 2))
    pump_n = rng.normal(30.0, 0.4, (80, 2))
    pump_a = rng.normal(30.8, 0.4, (40, 2))
    Xtr = np.vstack([fan_n, pump_n])
    tiptr = np.array(["fan"] * 80 + ["pump"] * 80)
    model = TipDedektoru().egit(Xtr, tiptr)
    Xte = np.vstack([fan_a, pump_a, pump_n[:20]])
    tipte = np.array(["fan"] * 40 + ["pump"] * 60)
    skor = model.anomali_skoru(Xte, tipte)
    pump = skor[40:]
    assert pump[:40].mean() > pump[40:].mean()


def test_tip_uyarlama_havuzu_geride_birakir():
    paket = _kaydirilmis_unite()
    havuz = degerlendir_diziler(*paket, kurulum="havuz")
    uyar = degerlendir_diziler(*paket, kurulum="tip_uyarlama")
    assert havuz["havuz_auc"] < 0.45
    assert uyar["makro_auc"] > 0.90
    assert "pr01" in uyar["isletme"]
    assert "f1" in uyar["isletme"]


def test_bilinmeyen_kurulum():
    paket = _kaydirilmis_unite()
    with pytest.raises(ValueError, match="kurulum"):
        degerlendir_diziler(*paket, kurulum="yok")
