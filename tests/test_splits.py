"""Test katmanı 0 — veri/split doğrulama, sızıntı kontrolleri."""
import pandas as pd
import pytest

from nabiz.data import splits as S


def test_birim_disi_sizinti_yok(man):
    df = S.birim_disi_split(man, test_orani=0.25, val_orani=0.125, seed=0)
    S.sizinti_yok(df, "makine_birimi")  # exception atmamalı
    g = df.groupby("makine_birimi")["bolme"].nunique()
    assert (g == 1).all(), "her birim tam bir bölmede olmalı"


def test_birim_disi_uc_bolme_de_dolu(man):
    df = S.birim_disi_split(man, seed=0)
    assert set(df["bolme"]) == {"train", "val", "test"}


def test_birim_disi_tabaka_korunur(man):
    df = S.birim_disi_split(man, seed=2, tabaka_sutunu="makine_tipi")
    for tip in man["makine_tipi"].unique():
        alt = df[df["makine_tipi"] == tip]
        assert alt["bolme"].nunique() >= 2, f"{tip} tek bölmede kalmış"


def test_birim_disi_deterministik(man):
    a = S.birim_disi_split(man, seed=3)
    b = S.birim_disi_split(man.sample(frac=1, random_state=42), seed=3)
    birlesim = a[["dosya", "bolme"]].merge(b[["dosya", "bolme"]], on="dosya")
    assert (birlesim["bolme_x"] == birlesim["bolme_y"]).all()


def test_birim_ici_anomali_egitimde_yok(man):
    df = S.birim_ici_split(man, sadece_normal_egitim=True, seed=0)
    egitim = df[df["bolme"].isin(["train", "val"])]
    assert (egitim["etiket"] == "normal").all()
    assert (df.loc[df["etiket"] == "anomali", "bolme"] == "test").all()


def test_birim_ici_dosya_sizintisi_yok(man):
    df = S.birim_ici_split(man, seed=0)
    S.dosya_sizintisi_yok(df)
    assert (df.groupby("dosya")["bolme"].nunique() == 1).all()


def test_birim_ici_deterministik_siradan_bagimsiz(man):
    a = S.birim_ici_split(man, seed=7)
    b = S.birim_ici_split(man.sample(frac=1, random_state=99), seed=7)
    birlesim = a[["dosya", "bolme"]].merge(b[["dosya", "bolme"]], on="dosya")
    assert (birlesim["bolme_x"] == birlesim["bolme_y"]).all()


def test_birim_ici_oranlar_yaklasik(man):
    df = S.birim_ici_split(man, test_orani=0.2, val_orani=0.1, seed=0)
    normaller = df[df["etiket"] == "normal"]
    pay = normaller["bolme"].value_counts(normalize=True)
    assert 0.13 < pay.get("test", 0) < 0.27
    assert 0.05 < pay.get("val", 0) < 0.16


def test_sizinti_tespiti_calisir():
    bozuk = pd.DataFrame(
        {
            "dosya": ["a", "b", "c", "d"],
            "makine_birimi": ["u1", "u1", "u2", "u2"],
            "etiket": ["normal"] * 4,
            "rejim": ["0dB"] * 4,
            "bolme": ["train", "test", "train", "train"],  # u1 hem train hem test
        }
    )
    with pytest.raises(S.SizintiHatasi):
        S.sizinti_yok(bozuk, "makine_birimi")


def test_sizinti_yok_bolme_sutunu_eksikse_hata(man):
    with pytest.raises(ValueError):
        S.sizinti_yok(man, "makine_birimi")


def test_rejim_holdout_hedef_domain_test(man):
    df = S.rejim_holdout_split(man, holdout_rejimler=["6dB"], seed=0)
    assert (df.loc[df["rejim"] == "6dB", "bolme"] == "test").all()
    assert "6dB" not in set(df.loc[df["bolme"] == "train", "rejim"])
    assert "6dB" not in set(df.loc[df["bolme"] == "val", "rejim"])


def test_rejim_holdout_bilinmeyen_rejim_hata(man):
    with pytest.raises(ValueError):
        S.rejim_holdout_split(man, holdout_rejimler=["yokboyle"], seed=0)


def test_manifest_parmak_izi_kararli(man):
    df = S.birim_disi_split(man, seed=0)
    h1 = S.manifest_parmak_izi(df)
    h2 = S.manifest_parmak_izi(df.sample(frac=1, random_state=5))
    assert h1 == h2
    assert len(h1) == 64


def test_manifest_parmak_izi_bolme_degisince_degisir(man):
    a = S.manifest_parmak_izi(S.birim_disi_split(man, seed=0))
    b = S.manifest_parmak_izi(S.birim_disi_split(man, seed=1))
    assert a != b


def test_bolme_ozeti_sekli_ve_toplamlar(man):
    df = S.birim_disi_split(man, seed=0)
    ozet = S.bolme_ozeti(df)
    assert "birim_sayisi" in ozet.columns
    assert ozet["birim_sayisi"].sum() == man["makine_birimi"].nunique()
    assert ozet["toplam"].sum() == len(df)
