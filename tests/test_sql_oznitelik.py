"""Test katmanı 0/2 — C-MAPSS SQL veri hattı: şema, yükleme, pencere-fonksiyonu öznitelikleri.

SQLite üzerinde koşar (CI'da PostgreSQL yok); aynı sorgu PostgreSQL'de
`09_rul_sql.py` içinde ayrıca pandas referansına karşı doğrulanır.
"""
import numpy as np
import pandas as pd
import pytest

from nabiz.data import cmapss as C
from nabiz.data import sql_oznitelik as S

SENSORLER = ["sensor_2", "sensor_7", "sensor_11"]


@pytest.fixture
def veri():
    return C.sentetik_uret(n_birim=6, min_cevrim=35, maks_cevrim=70, seed=4)


@pytest.fixture
def db(veri):
    b = S.baglan("sqlite://:memory:")
    S.sema_olustur(b)
    S.yukle(b, veri, "sentetik-train")
    yield b
    b.kapat()


def test_yukle_satir_sayisi_ve_idempotent(db, veri):
    assert S.satir_sayilari(db) == {"sentetik-train": len(veri)}
    S.yukle(db, veri, "sentetik-train")  # tekrar yükleme çoğaltmamalı
    assert S.satir_sayilari(db) == {"sentetik-train": len(veri)}


def test_yukle_eksik_sutun_reddeder(db, veri):
    with pytest.raises(ValueError, match="eksik"):
        S.yukle(db, veri.drop(columns=["sensor_3"]), "x")


def test_sql_oznitelikleri_pandas_referansiyla_esit(db, veri):
    sql = S.oznitelik_cek(db, SENSORLER, "sentetik-train")
    ref = S.oznitelik_pandas(veri, SENSORLER)
    assert len(sql) == len(ref) == len(veri)
    np.testing.assert_array_equal(sql["birim"].to_numpy(), ref["birim"].to_numpy())
    np.testing.assert_array_equal(sql["cevrim"].to_numpy(), ref["cevrim"].to_numpy())
    for ad in S.oznitelik_adlari(SENSORLER)[1:]:
        np.testing.assert_allclose(
            sql[ad].to_numpy(), ref[ad].to_numpy(), rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=ad
        )


def test_ilk_cevrimde_egim_ve_varyans_tanimsiz(db):
    sql = S.oznitelik_cek(db, SENSORLER, "sentetik-train")
    ilk = sql[sql["cevrim"] == 1]
    assert ilk["sensor_2_egim10"].isna().all()
    assert ilk["sensor_2_var30"].isna().all()
    assert ilk["sensor_2_sapma"].abs().max() == pytest.approx(0.0)  # ort10 == taban == x


def test_egim_dogrusal_seride_tam(veri):
    df = veri[veri["birim"] == 1].copy()
    df["sensor_2"] = 5.0 + 0.25 * df["cevrim"]  # eğim tam 0.25
    b = S.baglan("sqlite://:memory:")
    S.sema_olustur(b)
    S.yukle(b, df, "dogrusal")
    sql = S.oznitelik_cek(b, ["sensor_2"], "dogrusal")
    np.testing.assert_allclose(sql["sensor_2_egim10"].iloc[1:], 0.25, rtol=1e-9)
    np.testing.assert_allclose(sql["sensor_2_egim30"].iloc[1:], 0.25, rtol=1e-9)
    b.kapat()


def test_oznitelikler_nedensel_gelecek_satirlardan_etkilenmez(veri):
    """Bir motorun kesildiği noktadaki öznitelikler, sonrasındaki çevrimlerden bağımsız olmalı
    — test motorunun son çevriminde tahmin yapılırken sızıntı olmaması bunun sonucu."""
    tam = veri[veri["birim"] == 2]
    kesik = tam[tam["cevrim"] <= 25]
    b = S.baglan("sqlite://:memory:")
    S.sema_olustur(b)
    S.yukle(b, tam, "tam")
    S.yukle(b, kesik, "kesik")
    a = S.oznitelik_cek(b, SENSORLER, "tam")
    k = S.oznitelik_cek(b, SENSORLER, "kesik")
    pd.testing.assert_frame_equal(
        a[a["cevrim"] <= 25].drop(columns="kaynak").reset_index(drop=True),
        k.drop(columns="kaynak").reset_index(drop=True),
    )
    b.kapat()


def test_yalniz_son_cevrim_ve_birim_filtresi(db, veri):
    son = S.oznitelik_cek(db, SENSORLER, "sentetik-train", yalniz_son_cevrim=True)
    beklenen = veri.groupby("birim")["cevrim"].max()
    assert list(son["birim"]) == list(beklenen.index)
    assert list(son["cevrim"]) == list(beklenen.to_numpy())
    tek = S.oznitelik_cek(db, SENSORLER, "sentetik-train", birim=3, yalniz_son_cevrim=True)
    assert len(tek) == 1 and int(tek["birim"].iloc[0]) == 3


def test_test_rul_son_tablosu(veri):
    df = veri.copy()
    df["rul_test_sonu"] = df["birim"] * 10.0
    b = S.baglan("sqlite://:memory:")
    S.sema_olustur(b)
    S.yukle(b, df, "t")
    rul = S.test_rul_son(b, "t")
    assert rul.to_dict() == {i: i * 10.0 for i in range(1, 7)}
    b.kapat()


def test_bilinmeyen_sensor_sql_e_gomulmez():
    with pytest.raises(ValueError, match="bilinmeyen"):
        S.oznitelik_sorgusu(["sensor_2; DROP TABLE cmapss_olcum"])
    with pytest.raises(ValueError):
        S.oznitelik_sorgusu([])


def test_bilgilendirici_sensorler_sabitleri_eler(veri):
    df = veri.copy()
    df["sensor_1"] = 518.67  # sabit
    df["sensor_5"] = 14.62 + np.random.default_rng(0).normal(0, 1e-4, len(df))  # neredeyse sabit
    secilen = S.bilgilendirici_sensorler(df)
    assert "sensor_1" not in secilen and "sensor_5" not in secilen
    assert "sensor_2" in secilen


def test_oznitelik_adlari_sirasi():
    adlar = S.oznitelik_adlari(["sensor_2"])
    assert adlar == [
        "cevrim", "sensor_2_ort10", "sensor_2_ort30", "sensor_2_egim10",
        "sensor_2_egim30", "sensor_2_var30", "sensor_2_sapma",
    ]


def test_desteklenmeyen_dsn():
    with pytest.raises(ValueError, match="desteklenmeyen"):
        S.baglan("mysql://x")
