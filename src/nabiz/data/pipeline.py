"""Manifest + split üretimini `01_*` ve `02_*` betikleri arasında paylaş.

Deterministik: aynı veri + aynı seed → aynı bölme sütunları → aynı parmak izi.
"""
from __future__ import annotations

import pandas as pd

from nabiz import config
from nabiz.data import manifest as M
from nabiz.data import splits as S


def manifest_yukle() -> tuple[pd.DataFrame, str]:
    """Öncelik: gerçek MIMII > disk sentetik > bellek sentetik."""
    mimii_kok = config.HAM_VERI / "mimii"
    if mimii_kok.exists() and any(mimii_kok.rglob("*.wav")):
        return M.mimii_manifest(mimii_kok), "mimii"
    sentetik_kok = config.HAM_VERI / "sentetik"
    if sentetik_kok.exists() and any(sentetik_kok.rglob("*.wav")):
        return M.dizin_manifest(sentetik_kok, veri_seti="sentetik"), "sentetik-disk"
    return M.sentetik_manifest(seed=config.VARSAYILAN_SEED), "sentetik-bellek"


def split_stratejileri(man: pd.DataFrame, seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Uygulanabilir tüm split stratejilerini üret (rejim_holdout yalnız >1 rejimde)."""
    seed = config.VARSAYILAN_SEED if seed is None else seed
    strat = {
        "birim_ici": S.birim_ici_split(man, seed=seed),
        "birim_disi": S.birim_disi_split(man, seed=seed),
    }
    rejimler = sorted(man["rejim"].unique())
    if len(rejimler) > 1:
        strat["rejim_holdout"] = S.rejim_holdout_split(
            man, holdout_rejimler=[rejimler[-1]], seed=seed
        )
    return strat


def splitli_manifest(
    man: pd.DataFrame, seed: int | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Manifesti `bolme__<strateji>` sütunlarıyla genişlet + sızıntı kontrolü + parmak izleri."""
    birlesik = man.copy()
    parmak: dict[str, str] = {}
    for ad, df in split_stratejileri(man, seed).items():
        S.dosya_sizintisi_yok(df)
        if ad == "birim_disi":
            S.sizinti_yok(df, "makine_birimi")
        birlesik[f"bolme__{ad}"] = df[S.BOLME_SUTUNU].to_numpy()
        parmak[ad] = S.manifest_parmak_izi(df)
    return birlesik, parmak
