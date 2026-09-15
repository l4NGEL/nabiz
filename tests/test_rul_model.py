"""Test katmanı 2 — RUL regresörü (torch varsa)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from nabiz.models.rul import RULRegresor  # noqa: E402

N_SENSOR, PENCERE = 6, 12


@pytest.fixture
def veri():
    """RUL, tek kanaldaki doğrusal bozulma trendinin şiddetiyle ters orantılı
    (+ gürültü) — GRU'nun pencere sonu seviyesini okuması yeterli, kolay öğrenilir."""
    rng = np.random.default_rng(0)
    n = 500
    seviye = rng.uniform(0.0, 1.0, n)  # birim başına bozulma şiddeti (0..1)
    egilim = seviye[:, None] * np.linspace(0.0, 1.0, PENCERE)[None, :]  # (n, PENCERE) 0 -> seviye
    X = np.zeros((n, PENCERE, N_SENSOR), dtype=np.float32)
    X[:, :, 0] = egilim
    X += 0.05 * rng.normal(0, 1, (n, PENCERE, N_SENSOR))
    y = np.clip(125.0 - 100.0 * seviye + rng.normal(0, 3, n), 0, 125).astype(np.float32)
    return X.astype(np.float32), y


def test_rul_arayuz_ve_egitim(veri):
    X, y = veri
    model = RULRegresor(gizli=16, epok=15, seed=0).egit(X[:400], y[:400], X[400:], y[400:])
    assert len(model.gecmis_) > 0
    tahmin = model.tahmin(X[:10])
    assert tahmin.shape == (10,)
    assert np.all(np.isfinite(tahmin))


def test_rul_makul_korelasyon(veri):
    X, y = veri
    model = RULRegresor(gizli=32, epok=80, seed=0).egit(X[:400], y[:400], X[400:], y[400:])
    tahmin = model.tahmin(X[400:])
    korelasyon = np.corrcoef(tahmin, y[400:])[0, 1]
    assert korelasyon > 0.7


def test_rul_egitilmeden_tahmin_hata():
    with pytest.raises(RuntimeError):
        RULRegresor().tahmin(np.zeros((2, PENCERE, N_SENSOR), dtype=np.float32))


def test_rul_ornek_sayisi_uyusmazligi(veri):
    X, y = veri
    with pytest.raises(ValueError, match="uyuşmuyor"):
        RULRegresor(epok=2).egit(X[:100], y[:80])


def test_rul_deterministik(veri):
    X, y = veri
    a = RULRegresor(gizli=16, epok=10, seed=5).egit(X[:200], y[:200])
    b = RULRegresor(gizli=16, epok=10, seed=5).egit(X[:200], y[:200])
    np.testing.assert_allclose(a.tahmin(X[200:220]), b.tahmin(X[200:220]), rtol=1e-4)
