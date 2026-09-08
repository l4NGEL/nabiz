import numpy as np
import pytest

from nabiz import config
from nabiz.data import manifest as M


@pytest.fixture
def man():
    """8 birim × 3 tip, normal+anomali, iki rejim — sızıntı testleri için."""
    return M.sentetik_manifest(
        n_birim=8,
        n_normal=40,
        n_anomali=8,
        seed=1,
        tipler=("fan", "pump", "valve"),
        rejimler=("0dB", "6dB"),
    )


@pytest.fixture
def sr():
    return config.ORNEKLEME_HIZI


@pytest.fixture
def sinus(sr):
    def _yap(f0, sure_sn=2.0, genlik=1.0):
        t = np.arange(int(sure_sn * sr)) / sr
        return genlik * np.sin(2 * np.pi * f0 * t)

    return _yap
