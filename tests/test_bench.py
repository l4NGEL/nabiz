"""Test katmanı 4 — gecikme ölçümü (PR-02)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from nabiz.edge import bench as BN  # noqa: E402
from nabiz.edge import onnx_disa as O  # noqa: E402
from nabiz.models.autoencoder import DenseAE  # noqa: E402


@pytest.fixture
def oturum_ve_veri(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 16)).astype(np.float32)
    ae = DenseAE(gizli=(16,), dar=4, epok=3, seed=0).egit(X)
    yol = O.disa_aktar(ae._agi, X[:1], tmp_path / "m.onnx")
    return O.onnx_oturumu(yol), X


def test_gecikme_olc_alanlar_ve_sira(oturum_ve_veri):
    oturum, X = oturum_ve_veri
    r = BN.gecikme_olc(oturum, X, n=40, isinma=5)
    assert {"p50_ms", "p95_ms", "p99_ms", "ort_ms", "verim_ornek_sn", "n"} <= set(r)
    assert r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"]
    assert r["p50_ms"] > 0
    assert r["verim_ornek_sn"] > 0
    assert r["n"] == 40


def test_gecikme_olc_bos_veri_hata(oturum_ve_veri):
    oturum, _ = oturum_ve_veri
    with pytest.raises(ValueError):
        BN.gecikme_olc(oturum, np.zeros((0, 16), dtype=np.float32), n=5, isinma=1)


def test_gecikme_olc_deterministik_ornekleme(oturum_ve_veri):
    oturum, X = oturum_ve_veri
    a = BN.gecikme_olc(oturum, X, n=20, isinma=2, seed=1)
    b = BN.gecikme_olc(oturum, X, n=20, isinma=2, seed=1)
    assert a["n"] == b["n"] == 20  # zamanlama değişir, örnekleme seed'i sabit
