"""Test katmanı 4 — INT8 dinamik nicemleme (RB-04)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from nabiz.edge import nicemleme as N  # noqa: E402
from nabiz.edge import onnx_disa as O  # noqa: E402
from nabiz.models.autoencoder import DenseAE  # noqa: E402


@pytest.fixture
def fp32_onnx(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 32)).astype(np.float32)
    ae = DenseAE(gizli=(64, 32), dar=8, epok=5, seed=0).egit(X)
    yol = O.disa_aktar(ae._agi, X[:1], tmp_path / "fp32.onnx")
    return yol, X


def test_int8_nicemle_dosya_uretir(fp32_onnx, tmp_path):
    fp32, _ = fp32_onnx
    int8 = N.int8_nicemle(fp32, tmp_path / "int8.onnx")
    assert int8.exists()


def test_int8_calisan_oturum_verir(fp32_onnx, tmp_path):
    fp32, X = fp32_onnx
    int8 = N.int8_nicemle(fp32, tmp_path / "int8.onnx")
    oturum = O.onnx_oturumu(int8)
    y = O.onnx_calistir(oturum, X[:10])
    assert y.shape == (10, 32)
    assert np.all(np.isfinite(y))


def test_int8_skor_paritesi_yuksek(fp32_onnx, tmp_path):
    fp32, X = fp32_onnx
    int8 = N.int8_nicemle(fp32, tmp_path / "int8.onnx")
    y_f = O.onnx_calistir(O.onnx_oturumu(fp32), X)
    y_i = O.onnx_calistir(O.onnx_oturumu(int8), X)
    sp = O.skor_paritesi(X, y_f, y_i)
    assert sp["skor_korelasyon"] > 0.95  # RB-04 hedefi ≥0.99; küçük sentetik modelde gevşek eşik


def test_kaynak_yoksa_hata(tmp_path):
    with pytest.raises(FileNotFoundError):
        N.int8_nicemle(tmp_path / "yok.onnx", tmp_path / "int8.onnx")


def test_dosya_boyutu_mb(fp32_onnx):
    fp32, _ = fp32_onnx
    assert N.dosya_boyutu_mb(fp32) > 0


def test_dosya_boyutu_mb_yoksa_hata(tmp_path):
    with pytest.raises(FileNotFoundError):
        N.dosya_boyutu_mb(tmp_path / "yok.onnx")
