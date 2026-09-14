"""Test katmanı 2 — füzyon autoencoder (spektrogram ⊕ telemetri, torch varsa)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.models.fusion import FusionAE  # noqa: E402

N_MEL, T, N_TEL = 16, 24, 6


@pytest.fixture
def veri():
    rng = np.random.default_rng(0)
    taban_spek = rng.normal(0, 1, (1, N_MEL, 4)) @ rng.normal(0, 1, (1, 4, T))
    taban_tel = rng.normal(0, 1, N_TEL)

    Xs_n = np.repeat(taban_spek, 300, 0) + 0.1 * rng.normal(0, 1, (300, N_MEL, T))
    Xt_n = np.tile(taban_tel, (300, 1)) + 0.1 * rng.normal(0, 1, (300, N_TEL))

    Xs_a = np.repeat(taban_spek, 100, 0) + 0.1 * rng.normal(0, 1, (100, N_MEL, T))
    Xt_a = np.tile(taban_tel, (100, 1)) + 0.1 * rng.normal(0, 1, (100, N_TEL))
    Xt_a[:, :2] += 3.0  # anomali yalnız telemetri dalında belirgin

    return (
        Xs_n.astype(np.float32), Xt_n.astype(np.float32),
        Xs_a.astype(np.float32), Xt_a.astype(np.float32),
    )


def test_fusion_arayuz_ve_egitim(veri):
    Xs_n, Xt_n, _, _ = veri
    model = FusionAE(epok=8, seed=0).egit(Xs_n[:250], Xt_n[:250], Xs_n[250:], Xt_n[250:])
    assert len(model.gecmis_) > 0
    s = model.anomali_skoru(Xs_n[:10], Xt_n[:10])
    assert s.shape == (10,) and np.all(np.isfinite(s))


def test_fusion_telemetri_anomalisini_yakalar(veri):
    Xs_n, Xt_n, Xs_a, Xt_a = veri
    model = FusionAE(epok=60, tel_agirlik=0.5, seed=0).egit(Xs_n[:250], Xt_n[:250], Xs_n[250:], Xt_n[250:])
    s = np.r_[model.anomali_skoru(Xs_n[:100], Xt_n[:100]), model.anomali_skoru(Xs_a, Xt_a)]
    y = np.r_[np.zeros(100), np.ones(len(Xs_a))].astype(int)
    assert MT.auc(y, s) > 0.85


def test_fusion_dallar_toplami_tutarli(veri):
    Xs_n, Xt_n, Xs_a, Xt_a = veri
    model = FusionAE(epok=10, tel_agirlik=0.4, seed=0).egit(Xs_n[:250], Xt_n[:250])
    d = model.anomali_skoru_dallar(Xs_a, Xt_a)
    assert set(d) == {"toplam", "spektrogram", "telemetri"}
    np.testing.assert_allclose(d["toplam"], d["spektrogram"] + 0.4 * d["telemetri"], rtol=1e-4)


def test_fusion_ornek_sayisi_uyusmazligi(veri):
    Xs_n, Xt_n, _, _ = veri
    with pytest.raises(ValueError, match="uyuşmuyor"):
        FusionAE(epok=2).egit(Xs_n[:100], Xt_n[:80])


def test_fusion_egitilmeden_skor_hata():
    with pytest.raises(RuntimeError):
        FusionAE().anomali_skoru(np.zeros((2, N_MEL, T), dtype=np.float32), np.zeros((2, N_TEL), dtype=np.float32))


def test_fusion_deterministik(veri):
    Xs_n, Xt_n, Xs_a, Xt_a = veri
    a = FusionAE(epok=8, seed=5).egit(Xs_n[:200], Xt_n[:200])
    b = FusionAE(epok=8, seed=5).egit(Xs_n[:200], Xt_n[:200])
    np.testing.assert_allclose(
        a.anomali_skoru(Xs_a, Xt_a), b.anomali_skoru(Xs_a, Xt_a), rtol=1e-4
    )
