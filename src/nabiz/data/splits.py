"""Veri bölme stratejileri ve sızıntı kontrolleri.

Stratejiler
-----------
birim_ici_split
    Her makine birimi train/val/test'e katkı verir; bir DOSYA asla iki bölmede
    olmaz. Anomali örnekleri (opsiyonel) yalnız test'te — gözetimsiz kurulum.
    Standart MIMII değerlendirmesi.

birim_disi_split
    Bütün makine birimleri tek bir bölmede; makine tipine göre tabakalı.
    Bir BİRİM asla iki bölmede olmaz — görülmemiş fiziksel üniteye genelleme.

rejim_holdout_split
    Seçili çalışma rejim(ler)i = hedef domain (test); kalan = kaynak (train/val).
    Domain-shift değerlendirmesi (RB-01). Aynı birim hem kaynak hem hedefte
    olabilir — kasıtlı; izole edilen değişken rejim, birim değil.

Atama deterministiktir: satır sırasından bağımsız, `dosya` yolunun SHA-1'ine
dayanır. Bölme sütunu: `bolme` ∈ {train, val, test}.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

BOLME_SUTUNU = "bolme"
_GECERLI_BOLMELER = ("train", "val", "test")


class SizintiHatasi(AssertionError):
    """Bir grup (dosya veya birim) birden çok bölmede görüldü."""


# --------------------------------------------------------------------------- #
# yardımcılar
# --------------------------------------------------------------------------- #
def _birim_araligi(yol: str, tuz: bytes) -> float:
    """Yola bağlı deterministik [0, 1) değeri."""
    h = hashlib.sha1(tuz + yol.encode("utf-8")).digest()[:8]
    return (int.from_bytes(h, "big") % 1_000_000) / 1_000_000.0


def sizinti_yok(df: pd.DataFrame, grup_sutunu: str) -> None:
    """Hiçbir grup değeri birden çok bölmede olmamalı. Aksi halde SizintiHatasi."""
    if BOLME_SUTUNU not in df.columns:
        raise ValueError(f"'{BOLME_SUTUNU}' sütunu yok — önce bir split uygula")
    sayim = df.groupby(grup_sutunu)[BOLME_SUTUNU].nunique()
    ihlal = sayim[sayim > 1]
    if len(ihlal):
        ornek = ", ".join(map(str, ihlal.index[:10]))
        raise SizintiHatasi(
            f"{len(ihlal)} grup birden çok bölmede ('{grup_sutunu}'): {ornek}"
        )


def dosya_sizintisi_yok(df: pd.DataFrame) -> None:
    """Hiçbir dosya yolu birden çok bölmede olmamalı."""
    sizinti_yok(df, "dosya")


# --------------------------------------------------------------------------- #
# stratejiler
# --------------------------------------------------------------------------- #
def birim_ici_split(
    manifest: pd.DataFrame,
    test_orani: float = 0.20,
    val_orani: float = 0.10,
    seed: int = 0,
    sadece_normal_egitim: bool = True,
) -> pd.DataFrame:
    """Her birim içinde dosyaları böl (hash tabanlı, deterministik)."""
    df = manifest.copy()
    tuz = f"birim_ici:{seed}".encode()

    def ata(satir: pd.Series) -> str:
        if sadece_normal_egitim and satir["etiket"] != "normal":
            return "test"
        r = _birim_araligi(satir["dosya"], tuz)
        if r < test_orani:
            return "test"
        if r < test_orani + val_orani:
            return "val"
        return "train"

    df[BOLME_SUTUNU] = df.apply(ata, axis=1)
    dosya_sizintisi_yok(df)
    df.attrs["split"] = "birim_ici"
    return df


def birim_disi_split(
    manifest: pd.DataFrame,
    test_orani: float = 0.20,
    val_orani: float = 0.10,
    seed: int = 0,
    grup_sutunu: str = "makine_birimi",
    tabaka_sutunu: str | None = "makine_tipi",
) -> pd.DataFrame:
    """Bütün birimleri bölmelere ayır — birim, bölme sınırını geçmez."""
    df = manifest.copy()
    rng = np.random.default_rng(seed)
    df[BOLME_SUTUNU] = "train"

    if tabaka_sutunu is None:
        tabakalar: list = [None]
    else:
        tabakalar = sorted(df[tabaka_sutunu].unique())

    for tab in tabakalar:
        alt = df if tab is None else df[df[tabaka_sutunu] == tab]
        birimler = sorted(alt[grup_sutunu].unique())
        rng.shuffle(birimler)
        n = len(birimler)
        if n <= 1:
            continue  # tek birim → train'de kalır
        n_test = max(1, round(n * test_orani))
        kalan = n - n_test
        n_val = max(1, round(n * val_orani)) if kalan > 1 else 0
        test_b = set(birimler[:n_test])
        val_b = set(birimler[n_test : n_test + n_val])
        df.loc[df[grup_sutunu].isin(test_b), BOLME_SUTUNU] = "test"
        df.loc[df[grup_sutunu].isin(val_b), BOLME_SUTUNU] = "val"

    sizinti_yok(df, grup_sutunu)
    df.attrs["split"] = "birim_disi"
    return df


def rejim_holdout_split(
    manifest: pd.DataFrame,
    holdout_rejimler: list[str],
    val_orani: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Holdout rejim(ler)i = hedef domain (test); kalan = kaynak (train/val)."""
    df = manifest.copy()
    holdout = set(holdout_rejimler)
    mevcut = set(df["rejim"].unique())
    if not holdout & mevcut:
        raise ValueError(
            f"holdout rejimleri manifestte yok: {sorted(holdout)} ∉ {sorted(mevcut)}"
        )
    hedef_maske = df["rejim"].isin(holdout)
    df[BOLME_SUTUNU] = "train"
    df.loc[hedef_maske, BOLME_SUTUNU] = "test"

    tuz = f"rejim_holdout:{seed}".encode()
    kaynak_idx = df.index[~hedef_maske]
    for i in kaynak_idx:
        if _birim_araligi(df.at[i, "dosya"], tuz) < val_orani:
            df.at[i, BOLME_SUTUNU] = "val"

    dosya_sizintisi_yok(df)
    df.attrs["split"] = "rejim_holdout"
    df.attrs["holdout_rejimler"] = sorted(holdout)
    return df


# --------------------------------------------------------------------------- #
# özet / izlenebilirlik
# --------------------------------------------------------------------------- #
def bolme_ozeti(df: pd.DataFrame) -> pd.DataFrame:
    """Bölme × etiket kırılımı + bölme başına birim sayısı."""
    ozet = df.groupby([BOLME_SUTUNU, "etiket"]).size().unstack(fill_value=0)
    for et in ("normal", "anomali"):
        if et not in ozet.columns:
            ozet[et] = 0
    ozet["toplam"] = ozet.sum(axis=1)
    ozet["birim_sayisi"] = df.groupby(BOLME_SUTUNU)["makine_birimi"].nunique()
    # sabit bölme sırası
    return ozet.reindex([b for b in _GECERLI_BOLMELER if b in ozet.index])


def manifest_parmak_izi(df: pd.DataFrame) -> str:
    """Sıralı (dosya, bölme) çiftlerinin SHA-256'sı.

    Rapor ve eğitim koşuları bunu alıntılar; CI değişmediğini doğrular
    (kasıtlı güncelleme dışında).
    """
    if BOLME_SUTUNU not in df.columns:
        raise ValueError(f"'{BOLME_SUTUNU}' sütunu yok")
    govde = "\n".join(
        f"{d}\t{b}"
        for d, b in sorted(zip(df["dosya"], df[BOLME_SUTUNU], strict=True))
    )
    return hashlib.sha256(govde.encode("utf-8")).hexdigest()
