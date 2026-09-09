"""Test katmanı 0 — manifest (disk) + pipeline paylaşımlı üretim."""
import pandas as pd
import pytest

from nabiz.data import manifest as M
from nabiz.data import pipeline as P
from nabiz.data import splits as S


@pytest.fixture
def sahte_dizin(tmp_path):
    """MIMII benzeri düzende boş .wav dosyaları (yalnız yol; içerik okunmaz)."""
    for snr in ("0dB", "6dB"):
        for tip in ("fan", "pump"):
            for bidx in range(3):
                for durum, adet in (("normal", 5), ("abnormal", 2)):
                    d = tmp_path / snr / tip / f"id_{bidx:02d}" / durum
                    d.mkdir(parents=True)
                    for i in range(adet):
                        (d / f"{i:06d}.wav").touch()
    return tmp_path


def test_dizin_manifest_semasi(sahte_dizin):
    man = M.dizin_manifest(sahte_dizin, veri_seti="test")
    assert set(M.SUTUNLAR).issubset(man.columns)
    assert man["veri_seti"].eq("test").all()
    assert man["makine_birimi"].nunique() == 2 * 3  # tip x birim
    assert set(man["rejim"]) == {"0dB", "6dB"}
    assert set(man["etiket"]) == {"normal", "anomali"}
    # dosya yolu göreli, '/' ayraçlı
    assert not man["dosya"].str.contains(r"\\").any()
    assert man.attrs["veri_kok"]


def test_dizin_manifest_bos_dizin_hata(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.dizin_manifest(tmp_path)


def test_mutlak_yollar_geri_cozer(sahte_dizin):
    man = M.dizin_manifest(sahte_dizin)
    yollar = M.mutlak_yollar(man)
    assert len(yollar) == len(man)
    assert all(p.exists() for p in yollar)


def test_splitli_manifest_sutunlar_ve_sizinti(sahte_dizin):
    man = M.dizin_manifest(sahte_dizin)
    birlesik, parmak = P.splitli_manifest(man, seed=1)
    assert {"bolme__birim_ici", "bolme__birim_disi", "bolme__rejim_holdout"}.issubset(birlesik.columns)
    assert set(parmak) == {"birim_ici", "birim_disi", "rejim_holdout"}
    assert all(len(h) == 64 for h in parmak.values())
    # birim_disi sızıntısız
    kontrol = birlesik[["makine_birimi", "bolme__birim_disi"]].rename(
        columns={"bolme__birim_disi": "bolme"}
    )
    S.sizinti_yok(kontrol, "makine_birimi")


def test_splitli_manifest_deterministik(sahte_dizin):
    man = M.dizin_manifest(sahte_dizin)
    _, p1 = P.splitli_manifest(man, seed=7)
    _, p2 = P.splitli_manifest(man.sample(frac=1, random_state=3), seed=7)
    assert p1 == p2


def test_split_stratejileri_tek_rejimde_holdout_yok(tmp_path):
    d = tmp_path / "fan" / "id_00" / "normal"
    d.mkdir(parents=True)
    for i in range(6):
        (d / f"{i:06d}.wav").touch()
    (tmp_path / "fan" / "id_01" / "normal").mkdir(parents=True)
    for i in range(6):
        (tmp_path / "fan" / "id_01" / "normal" / f"{i:06d}.wav").touch()
    man = M.dizin_manifest(tmp_path)
    assert man["rejim"].nunique() == 1
    strat = P.split_stratejileri(man)
    assert "rejim_holdout" not in strat


def test_manifest_dogrula_bozuk_etiket():
    kotu = pd.DataFrame(
        {c: ["x"] for c in M.SUTUNLAR} | {"etiket": ["gecersiz"]}
    )
    with pytest.raises(ValueError):
        M.manifest_dogrula(kotu)
