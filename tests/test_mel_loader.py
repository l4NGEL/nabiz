"""Test katmanı 0/2 — mel.npz + parquet -> split bazında tensör."""
import numpy as np
import pandas as pd
import pytest

from nabiz.data import mel as MEL


@pytest.fixture
def mel_cache(tmp_path):
    rng = np.random.default_rng(0)
    n, n_mel, t = 120, 16, 20
    df = pd.DataFrame(
        {
            "makine_tipi": rng.choice(["fan", "pump"], n),
            "makine_birimi": "fan/id_00",
            "rejim": "0dB",
            "ariza_tipi": "yok",
            "etiket": rng.choice(["normal", "anomali"], n, p=[0.7, 0.3]),
            "bolme__birim_disi": rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2]),
        }
    )
    X = rng.normal(0, 1, (n, n_mel, t)).astype(np.float32)
    (tmp_path / "oznitelikler.parquet").write_bytes(b"")
    df.to_parquet(tmp_path / "oznitelikler.parquet", index=False)
    np.savez_compressed(tmp_path / "mel.npz", X=X)
    return tmp_path / "oznitelikler.parquet", tmp_path / "mel.npz"


def test_yukle_sekiller(mel_cache):
    p, n = mel_cache
    v = MEL.yukle("birim_disi", parquet=p, npz=n)
    assert v.sekil == (16, 20)
    assert v.X_train.ndim == 3
    assert v.X_train.shape[0] == v.y_train.shape[0]
    assert v.X_test.shape[0] == len(v.meta_test)
    assert v.X_train.dtype == np.float32


def test_train_normal_alt_kume(mel_cache):
    p, n = mel_cache
    v = MEL.yukle("birim_disi", parquet=p, npz=n)
    assert v.X_train_normal.shape[0] == int((v.y_train == 0).sum())


def test_satir_sayisi_uyusmazligi_hata(tmp_path):
    df = pd.DataFrame(
        {
            "makine_tipi": ["fan"] * 10, "makine_birimi": ["fan/id_00"] * 10,
            "rejim": ["0dB"] * 10, "ariza_tipi": ["yok"] * 10,
            "etiket": ["normal"] * 10, "bolme__birim_disi": ["train"] * 10,
        }
    )
    df.to_parquet(tmp_path / "oznitelikler.parquet", index=False)
    np.savez_compressed(tmp_path / "mel.npz", X=np.zeros((7, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="uyuşmuyor"):
        MEL.yukle("birim_disi", parquet=tmp_path / "oznitelikler.parquet", npz=tmp_path / "mel.npz")


def test_npz_yoksa_hata(tmp_path):
    with pytest.raises(FileNotFoundError):
        MEL.yukle(npz=tmp_path / "yok.npz", parquet=tmp_path / "yok.parquet")


def test_bilinmeyen_strateji(mel_cache):
    p, n = mel_cache
    with pytest.raises(ValueError, match="split yok"):
        MEL.yukle("yokboyle", parquet=p, npz=n)
