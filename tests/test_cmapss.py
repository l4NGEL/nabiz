"""Test katmanı 0/2 — C-MAPSS veri yükleme, sentetik üretim, pencereleme."""
import numpy as np
import pandas as pd
import pytest

from nabiz.data import cmapss as C


@pytest.fixture
def sentetik():
    return C.sentetik_uret(n_birim=12, min_cevrim=40, maks_cevrim=80, seed=0)


def test_sentetik_uret_sema(sentetik):
    assert list(sentetik.columns) == C.SUTUNLAR
    assert sentetik["birim"].nunique() == 12
    assert sentetik.attrs["kaynak"] == "cmapss-sentetik"
    assert len(sentetik.attrs["bilgilendirici_sensorler"]) == 8


def test_sentetik_uret_cevrim_run_to_failure(sentetik):
    # her birim 1'den başlayıp kesintisiz artmalı
    for _birim, grup in sentetik.groupby("birim"):
        cevrimler = grup.sort_values("cevrim")["cevrim"].to_numpy()
        np.testing.assert_array_equal(cevrimler, np.arange(1, len(cevrimler) + 1))


def test_sentetik_uret_deterministik():
    a = C.sentetik_uret(n_birim=5, seed=3)
    b = C.sentetik_uret(n_birim=5, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_birim_split_kapsayici_ve_deterministik(sentetik):
    birimler = sentetik["birim"].to_numpy()
    a = C.birim_split(birimler, seed=0)
    b = C.birim_split(birimler[::-1], seed=0)
    assert a == b
    assert set(a) == set(birimler.tolist())
    assert set(a.values()) <= {"train", "val", "test"}


def test_pencerele_sekil_ve_rul_araligi(sentetik):
    X, y, bid = C.pencerele(sentetik, pencere=15)
    assert X.ndim == 3 and X.shape[1] == 15 and X.shape[2] == 21
    assert X.shape[0] == y.shape[0] == bid.shape[0]
    assert np.all(y >= 0) and np.all(y <= C.TAVAN_RUL)


def test_pencerele_kisa_birim_atlanir():
    df = pd.DataFrame(
        {"birim": [1, 1, 1], "cevrim": [1, 2, 3], **{f"ayar_{i}": [0.0] * 3 for i in range(1, 4)},
         **{f"sensor_{i}": [0.0] * 3 for i in range(1, 22)}}
    )
    with pytest.raises(ValueError, match="pencere küçült"):
        C.pencerele(df, pencere=10)


def test_pencerele_son_rul_dogru(sentetik):
    # bir birimin son penceresindeki RUL ~0 olmalı (run-to-failure, rul_test_sonu yok)
    X, y, bid = C.pencerele(sentetik, pencere=15, tavan_rul=1000)  # tavansız gerçek değeri gör
    for b in np.unique(bid):
        idx = np.where(bid == b)[0]
        assert y[idx].min() == pytest.approx(0.0)  # son pencerede RUL 0


def test_son_pencere_birim_basina_tek_satir(sentetik):
    X, y, bid = C.son_pencere(sentetik, pencere=15)
    assert len(np.unique(bid)) == len(bid)  # birim başına 1 örnek
    assert X.shape[1:] == (15, 21)


def test_son_pencere_train_rul_sifir(sentetik):
    _X, y, _bid = C.son_pencere(sentetik, pencere=15)
    assert np.allclose(y, 0.0)  # run-to-failure train verisinde son nokta = arıza


def test_train_oku_gercek_format(tmp_path):
    satirlar = []
    for birim in (1, 2):
        for cevrim in range(1, 6):
            deger = [birim, cevrim, 0.1, 0.2, 0.3] + list(range(21))
            satirlar.append(" ".join(map(str, deger)))
    yol = tmp_path / "train_FD001.txt"
    yol.write_text("\n".join(satirlar), encoding="utf-8")
    df = C.train_oku(yol)
    assert list(df.columns) == C.SUTUNLAR
    assert len(df) == 10
    assert df.attrs["kaynak"] == "cmapss-gercek-train"


def test_test_oku_rul_birlestirir(tmp_path):
    satirlar = [" ".join(map(str, [1, c, 0.1, 0.2, 0.3] + list(range(21)))) for c in range(1, 4)]
    (tmp_path / "test_FD001.txt").write_text("\n".join(satirlar), encoding="utf-8")
    (tmp_path / "RUL_FD001.txt").write_text("42", encoding="utf-8")
    df = C.test_oku(tmp_path / "test_FD001.txt", tmp_path / "RUL_FD001.txt")
    assert (df["rul_test_sonu"] == 42).all()
