"""Test katmanı 0/2 — öznitelik parquet -> model-hazır diziler."""
import numpy as np
import pandas as pd
import pytest

from nabiz.data import oznitelik as OZ


@pytest.fixture
def parquet(tmp_path):
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame(
        {
            "makine_tipi": rng.choice(["fan", "pump"], n),
            "makine_birimi": rng.choice(["fan/id_00", "pump/id_00"], n),
            "rejim": rng.choice(["0dB", "6dB"], n),
            "ariza_tipi": "yok",
            "etiket": rng.choice(["normal", "anomali"], n, p=[0.7, 0.3]),
            "bolme__birim_disi": rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2]),
        }
    )
    for i in range(5):
        df[f"a_mel_ort_{i:02d}"] = rng.normal(0, 1, n)
    df["a_rms"] = rng.random(n)
    df["z_tepe"] = rng.random(n)
    df["t_rpm_ort"] = rng.normal(1800, 20, n)
    yol = tmp_path / "oznitelikler.parquet"
    df.to_parquet(yol, index=False)
    return yol


def test_yukle_sekiller_ve_etiket(parquet):
    v = OZ.yukle("birim_disi", parquet=parquet)
    assert v.X_train.shape[1] == len(v.oznitelik_adlari) == 8  # 5 mel + rms + z + t
    assert v.X_train.shape[0] == v.y_train.shape[0]
    assert set(np.unique(v.y_test)) <= {0, 1}
    assert len(v.meta_test) == v.X_test.shape[0]


def test_yukle_train_normal_alt_kume(parquet):
    v = OZ.yukle("birim_disi", parquet=parquet)
    assert v.X_train_normal.shape[0] == int((v.y_train == 0).sum())
    assert v.X_train_normal.shape[0] < v.X_train.shape[0]


def test_yukle_grup_secimi_ablasyon(parquet):
    v = OZ.yukle("birim_disi", parquet=parquet, gruplar=("a",))
    assert all(a.startswith("a_") for a in v.oznitelik_adlari)
    assert not any(a.startswith(("z_", "t_")) for a in v.oznitelik_adlari)


def test_yukle_bilinmeyen_strateji(parquet):
    with pytest.raises(ValueError, match="split yok"):
        OZ.yukle("yokboyle", parquet=parquet)


def test_yukle_parquet_yoksa(tmp_path):
    with pytest.raises(FileNotFoundError):
        OZ.yukle("birim_disi", parquet=tmp_path / "yok.parquet")


def test_grup_indeksleri(parquet):
    v = OZ.yukle("birim_disi", parquet=parquet)
    idx = OZ.grup_indeksleri(v.oznitelik_adlari, ("t",))
    assert idx == [v.oznitelik_adlari.index("t_rpm_ort")]
    assert len(OZ.grup_indeksleri(v.oznitelik_adlari, ("a", "z"))) == 7


def test_meta_train_val_hizali(parquet):
    v = OZ.yukle("birim_disi", parquet=parquet)
    assert len(v.meta_train) == v.X_train.shape[0]
    assert len(v.meta_val) == v.X_val.shape[0]
    assert "makine_birimi" in v.meta_train.columns


def test_kirilim_erisimi(parquet):
    v = OZ.yukle("birim_disi", parquet=parquet)
    tipler = v.kirilim("makine_tipi")
    assert len(tipler) == v.X_test.shape[0]
    assert set(tipler) <= {"fan", "pump"}
