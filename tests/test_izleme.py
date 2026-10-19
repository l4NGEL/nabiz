"""PCA artığı, düzlem dışındaki noktayı düzlem içindekinden yüksek skorlar."""
import numpy as np

from nabiz.models.izleme import TipSpe


def test_spe_duzlem_disini_ayirir():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[:, 2:] *= 0.01
    tip = np.array(["fan"] * 300)
    model = TipSpe(n_bilesen=2).egit(X, tip)
    ic = model.anomali_skoru(np.array([[0.2, -0.1, 0.0, 0.0]]), np.array(["fan"]))
    dis = model.anomali_skoru(np.array([[0.2, -0.1, 4.0, 4.0]]), np.array(["fan"]))
    assert dis[0] > ic[0] * 5
