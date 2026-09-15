"""Test katmanı 2 — OOD (dağılım-dışı) sağlamlık kontrolü."""
import numpy as np
import pytest

from nabiz.eval import ood as O


def test_gurultu_uret_sekil_ve_dtype():
    X = np.random.default_rng(0).normal(0, 1, (20, 8)).astype(np.float32)
    g = O.gurultu_uret(X, seed=1)
    assert g.shape == X.shape
    assert g.dtype == X.dtype
    assert not np.allclose(g, X)


def test_gurultu_uret_deterministik():
    X = np.random.default_rng(0).normal(0, 1, (20, 8))
    a, b = O.gurultu_uret(X, seed=5), O.gurultu_uret(X, seed=5)
    np.testing.assert_array_equal(a, b)


def test_karistir_uret_marjinal_korunur():
    X = np.random.default_rng(0).normal(0, 1, (50, 6))
    k = O.karistir_uret(X, seed=0)
    assert k.shape == X.shape
    for j in range(X.shape[1]):
        np.testing.assert_allclose(np.sort(k[:, j]), np.sort(X[:, j]))
    # ama satır-satır özdeş değil (korelasyon bozulmuş)
    assert not np.allclose(k, X)


def test_karistir_uret_3b_destekler():
    X = np.random.default_rng(0).normal(0, 1, (10, 4, 5))
    k = O.karistir_uret(X, seed=0)
    assert k.shape == X.shape


def test_ood_karsilastir_belirgin_ayrim():
    rng = np.random.default_rng(0)
    s_ic = rng.normal(0, 1, 200)
    s_ood = rng.normal(5, 1, 200)
    r = O.ood_karsilastir(s_ic, s_ood)
    assert r["ayrim_auc"] > 0.95
    assert r["ood_ort"] > r["ic_dagilim_ort"]
    assert r["ood_ic_dagilimdan_yuksek_oran"] > 0.9


def test_ood_karsilastir_ayirt_edilemez():
    rng = np.random.default_rng(0)
    s_ic = rng.normal(0, 1, 300)
    s_ood = rng.normal(0, 1, 300)  # aynı dağılım -> model OOD'u yakalayamıyor
    r = O.ood_karsilastir(s_ic, s_ood)
    assert 0.35 < r["ayrim_auc"] < 0.65


def test_ood_karsilastir_bos_hata():
    with pytest.raises(ValueError):
        O.ood_karsilastir(np.array([]), np.array([1.0]))
