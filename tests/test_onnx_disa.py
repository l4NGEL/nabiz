"""Test katmanı 4 — ONNX export + parite doğrulama (torch + onnxruntime varsa)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from nabiz.edge import onnx_disa as O  # noqa: E402
from nabiz.models.autoencoder import DenseAE  # noqa: E402
from nabiz.models.fusion import FusionAE  # noqa: E402


@pytest.fixture
def egitilmis_ae():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (150, 20)).astype(np.float32)
    return DenseAE(gizli=(16,), dar=4, epok=5, seed=0).egit(X), X


def test_disa_aktar_ve_oturum(egitilmis_ae, tmp_path):
    ae, X = egitilmis_ae
    yol = O.disa_aktar(ae._agi, X[:1], tmp_path / "model.onnx")
    assert yol.exists()
    oturum = O.onnx_oturumu(yol)
    assert oturum.get_inputs()[0].name


def test_onnx_calistir_sekil_dogru(egitilmis_ae, tmp_path):
    ae, X = egitilmis_ae
    oturum = O.onnx_oturumu(O.disa_aktar(ae._agi, X[:1], tmp_path / "m.onnx"))
    y = O.onnx_calistir(oturum, X[:16])
    assert y.shape == (16, 20)


def test_parite_yuksek_dogruluk(egitilmis_ae, tmp_path):
    ae, X = egitilmis_ae
    oturum = O.onnx_oturumu(O.disa_aktar(ae._agi, X[:1], tmp_path / "m.onnx"))
    p = O.pariteyi_dogrula(ae._agi, oturum, X[:32])
    assert p["gecti"]
    assert p["maks_mutlak_fark"] < 1e-3
    assert p["korelasyon"] > 0.999


def test_farkli_yigin_boyutu_calisir(egitilmis_ae, tmp_path):
    """dynamic_axes: export tek örnekle, çalıştırma farklı yığın boyutuyla."""
    ae, X = egitilmis_ae
    oturum = O.onnx_oturumu(O.disa_aktar(ae._agi, X[:1], tmp_path / "m.onnx"))
    for n in (1, 5, 33):
        y = O.onnx_calistir(oturum, X[:n])
        assert y.shape[0] == n


def test_skor_paritesi_ozdes_sifir():
    y = np.random.default_rng(0).normal(0, 1, (20, 10)).astype(np.float32)
    gercek = y + 0.1
    sp = O.skor_paritesi(gercek, y, y)
    assert sp["maks_skor_farki"] == 0.0
    assert sp["skor_korelasyon"] == pytest.approx(1.0)


def test_yeniden_yapilandirma_skoru_sekli():
    g = np.zeros((5, 4, 3))
    t = np.ones((5, 4, 3))
    s = O.yeniden_yapilandirma_skoru(g, t)
    assert s.shape == (5,)
    assert np.allclose(s, 1.0)


def test_dosya_boyutu_mb(egitilmis_ae, tmp_path):
    ae, X = egitilmis_ae
    yol = O.disa_aktar(ae._agi, X[:1], tmp_path / "m.onnx")
    assert O.dosya_boyutu_mb(yol) > 0


def test_coklu_girdi_fusion_export(tmp_path):
    rng = np.random.default_rng(0)
    Xs = rng.normal(0, 1, (60, 8, 10)).astype(np.float32)
    Xt = rng.normal(0, 1, (60, 5)).astype(np.float32)
    model = FusionAE(spek_kanallar=(4, 8, 16), spek_dar=6, tel_dar=3, ortak_dar=4, epok=3, seed=0)
    model.egit(Xs, Xt)

    class _SarmalayiciSpekCikti(torch.nn.Module):
        """ONNX export tek çıktı ister — spektrogram dalını raporla (parite testi için yeterli)."""

        def __init__(self, agi):
            super().__init__()
            self.agi = agi

        def forward(self, x_spek, x_tel):
            return self.agi(x_spek, x_tel)[0]

    sarmalayici = _SarmalayiciSpekCikti(model._agi)
    yol = O.disa_aktar(sarmalayici, (Xs[:1], Xt[:1]), tmp_path / "fusion.onnx", girdi_adlari=["spek", "tel"])
    oturum = O.onnx_oturumu(yol)
    p = O.pariteyi_dogrula(sarmalayici, oturum, Xs[:8], Xt[:8])
    assert p["gecti"]
