"""Test katmanı 2 — baseline anomali dedektörleri."""
import numpy as np
import pytest

from nabiz.eval import metrics as MT
from nabiz.models import baseline as B
from nabiz.models.temel import AnomaliModeli, Olcekleyici


@pytest.fixture
def ayrilabilir():
    """Normal ~ N(0, I); anomali kaymış. 12 öznitelik, 'a_rms' 0. indeks."""
    rng = np.random.default_rng(0)
    adlar = ["a_rms"] + [f"a_mel_ort_{i:02d}" for i in range(11)]
    X_normal = rng.normal(0.0, 1.0, (400, 12))
    X_normal[:, 0] = np.abs(rng.normal(0.3, 0.05, 400))  # a_rms ~ 0.3
    X_anom = rng.normal(0.0, 1.0, (120, 12))
    X_anom[:, 0] = np.abs(rng.normal(0.6, 0.1, 120))     # a_rms yüksek
    X_anom[:, 3:6] += 3.0                                 # bazı bantlar kaymış
    return adlar, X_normal, X_anom


@pytest.mark.parametrize("yap", ["rms", "iso", "maha"])
def test_baseline_ayirir(ayrilabilir, yap):
    adlar, X_normal, X_anom = ayrilabilir
    olc = Olcekleyici().egit(X_normal)
    if yap == "rms":
        model = B.RMSEsigi(adlar)
        Xn, Xa = X_normal, X_anom            # RMS ham ölçekte
    elif yap == "iso":
        model = B.IzolasyonOrmani(seed=0)
        Xn, Xa = olc.uygula(X_normal), olc.uygula(X_anom)
    else:
        model = B.Mahalanobis()
        Xn, Xa = olc.uygula(X_normal), olc.uygula(X_anom)

    model.egit(Xn)
    assert isinstance(model, AnomaliModeli)
    s = np.r_[model.anomali_skoru(Xn), model.anomali_skoru(Xa)]
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))].astype(int)
    assert MT.auc(y, s) > 0.75


def test_rms_esigi_bilinmeyen_oznitelik():
    with pytest.raises(ValueError):
        B.RMSEsigi(["a_mel_ort_00"], oznitelik="a_rms")


def test_baseline_nan_reddedilir(ayrilabilir):
    adlar, X_normal, _ = ayrilabilir
    X = X_normal.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        B.Mahalanobis().egit(X)


def test_olcekleyici_egit_uygula():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, (200, 4))
    olc = Olcekleyici()
    Z = olc.egit_uygula(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-6)


def test_olcekleyici_sifir_varyans_kanal():
    X = np.c_[np.ones(50), np.arange(50.0)]
    Z = Olcekleyici().egit_uygula(X)
    assert np.all(np.isfinite(Z))
