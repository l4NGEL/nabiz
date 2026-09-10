"""Test katmanı 2 — mel-tensör autoencoder'ları (torch varsa)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.models._egitim import TemelAE  # noqa: E402
from nabiz.models.cnn import ConvAE  # noqa: E402
from nabiz.models.crnn import CRNNAE  # noqa: E402
from nabiz.models.temel import AnomaliModeli  # noqa: E402
from nabiz.models.transformer import SpekTransformerAE  # noqa: E402

N_MEL, T = 16, 24


@pytest.fixture
def mel_veri():
    """Normal: düşük-ranklı zaman-frekans deseni. Anomali: + lokal darbe bloğu."""
    rng = np.random.default_rng(0)
    taban = rng.normal(0, 1, (1, N_MEL, 4)) @ rng.normal(0, 1, (1, 4, T))
    X_normal = np.repeat(taban, 400, 0) + 0.1 * rng.normal(0, 1, (400, N_MEL, T))
    X_anom = np.repeat(taban, 120, 0) + 0.1 * rng.normal(0, 1, (120, N_MEL, T))
    X_anom[:, 4:9, 8:14] += 3.0  # lokal anomali bloğu
    # bin bazında standardize (03_model_egit ile aynı)
    mu, sd = X_normal.mean((0, 2)), X_normal.std((0, 2))

    def z(a):
        return (a - mu[None, :, None]) / sd[None, :, None]

    return z(X_normal).astype(np.float32), z(X_anom).astype(np.float32)


@pytest.mark.parametrize("Yap", [ConvAE, CRNNAE, SpekTransformerAE])
def test_mel_ae_arayuz_ve_egitim(mel_veri, Yap):
    X_normal, _ = mel_veri
    ae = Yap(epok=6, seed=0).egit(X_normal[:350], X_val=X_normal[350:])
    assert isinstance(ae, TemelAE)
    assert isinstance(ae, AnomaliModeli)
    assert len(ae.gecmis_) > 0
    s = ae.anomali_skoru(X_normal[:8])
    assert s.shape == (8,) and np.all(np.isfinite(s))


@pytest.mark.parametrize("Yap", [ConvAE, CRNNAE, SpekTransformerAE])
def test_mel_ae_anomaliyi_ayirir(mel_veri, Yap):
    X_normal, X_anom = mel_veri
    ae = Yap(epok=60, seed=0).egit(X_normal[:350], X_val=X_normal[350:])
    s = np.r_[ae.anomali_skoru(X_normal[:120]), ae.anomali_skoru(X_anom)]
    y = np.r_[np.zeros(120), np.ones(len(X_anom))].astype(int)
    assert MT.auc(y, s) > 0.80


def test_mel_ae_egitilmeden_skor_hata():
    with pytest.raises(RuntimeError):
        ConvAE().anomali_skoru(np.zeros((2, N_MEL, T), dtype=np.float32))


def test_transformer_gecersiz_kafa():
    with pytest.raises(ValueError):
        SpekTransformerAE(d_model=64, kafa=5)


def test_conv_ae_deterministik(mel_veri):
    X_normal, X_anom = mel_veri
    a = ConvAE(epok=10, seed=3).egit(X_normal[:300])
    b = ConvAE(epok=10, seed=3).egit(X_normal[:300])
    np.testing.assert_allclose(a.anomali_skoru(X_anom), b.anomali_skoru(X_anom), rtol=1e-4)
