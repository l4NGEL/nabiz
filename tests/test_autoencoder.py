"""Test katmanı 2 — dense autoencoder (torch varsa)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.models.autoencoder import DenseAE  # noqa: E402
from nabiz.models.temel import AnomaliModeli, Olcekleyici  # noqa: E402


@pytest.fixture
def veri():
    """Normal: 5B manifold üzerinde. Anomali: manifold + büyük dik gürültü."""
    rng = np.random.default_rng(0)
    baz = rng.normal(0, 1, (20, 5))
    X_normal = rng.normal(0, 1, (500, 5)) @ baz.T + 0.02 * rng.normal(0, 1, (500, 20))
    X_anom = rng.normal(0, 1, (150, 5)) @ baz.T + 2.0 * rng.normal(0, 1, (150, 20))
    olc = Olcekleyici().egit(X_normal)
    return olc.uygula(X_normal), olc.uygula(X_anom)


def test_ae_egitim_ve_arayuz(veri):
    X_normal, _ = veri
    ae = DenseAE(gizli=(32, 16), dar=4, epok=40, seed=0).egit(X_normal[:400], X_val=X_normal[400:])
    assert isinstance(ae, AnomaliModeli)
    assert len(ae.gecmis_) > 0
    s = ae.anomali_skoru(X_normal[:10])
    assert s.shape == (10,) and np.all(np.isfinite(s))


def test_ae_anomaliyi_ayirir(veri):
    X_normal, X_anom = veri
    ae = DenseAE(gizli=(32, 16), dar=4, epok=150, seed=0).egit(X_normal[:450], X_val=X_normal[450:])
    s = np.r_[ae.anomali_skoru(X_normal[:150]), ae.anomali_skoru(X_anom)]
    y = np.r_[np.zeros(150), np.ones(len(X_anom))].astype(int)
    assert MT.auc(y, s) > 0.90


def test_ae_egitilmeden_skor_hata():
    with pytest.raises(RuntimeError):
        DenseAE().anomali_skoru(np.zeros((4, 10)))


def test_ae_erken_durma():
    # neredeyse sabit veri → AE hızla yakınsar → sabır tetiklenir
    rng = np.random.default_rng(0)
    X = np.ones((300, 12)) + 0.001 * rng.normal(0, 1, (300, 12))
    ae = DenseAE(gizli=(8,), dar=3, epok=300, sabir=6, seed=0).egit(X[:250], X_val=X[250:])
    assert len(ae.gecmis_) < 300


def test_ae_deterministik(veri):
    X_normal, X_anom = veri
    a = DenseAE(gizli=(16,), dar=4, epok=20, seed=7).egit(X_normal[:400])
    b = DenseAE(gizli=(16,), dar=4, epok=20, seed=7).egit(X_normal[:400])
    sa, sb = a.anomali_skoru(X_anom), b.anomali_skoru(X_anom)
    np.testing.assert_allclose(sa, sb, rtol=1e-4)
