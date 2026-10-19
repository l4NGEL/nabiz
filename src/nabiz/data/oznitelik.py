"""`oznitelikler.parquet` -> model-hazır diziler (split bazında).

Öznitelik grupları sütun önekiyle ayrılır: a_ (akustik), z_ (zarf), t_ (telemetri).
`gruplar` ile alt küme seçilebilir (ablasyon için).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nabiz import config

_ONEKLER = ("a_", "z_", "t_")


@dataclass
class BolmeVerisi:
    strateji: str
    oznitelik_adlari: list[str]
    X_train: np.ndarray          # tüm train örnekleri
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame = field(repr=False)   # makine_tipi, rejim, ariza_tipi, ...
    meta_train: pd.DataFrame = field(repr=False)
    meta_val: pd.DataFrame = field(repr=False)

    @property
    def X_train_normal(self) -> np.ndarray:
        return self.X_train[self.y_train == 0]

    def kirilim(self, sutun: str) -> np.ndarray:
        return self.meta_test[sutun].to_numpy()

    def ozet(self) -> str:
        return (
            f"[{self.strateji}] train {self.X_train.shape} "
            f"(normal {int((self.y_train == 0).sum())}), "
            f"val {self.X_val.shape}, test {self.X_test.shape} "
            f"(anomali {int(self.y_test.sum())}), "
            f"{len(self.oznitelik_adlari)} öznitelik"
        )


_META = ["makine_tipi", "makine_birimi", "rejim", "ariza_tipi"]
_META_OPSIYONEL = ["dosya", "pencere_idx"]


def _onekler(gruplar: tuple[str, ...] | None) -> tuple[str, ...]:
    return tuple(g if g.endswith("_") else g + "_" for g in (gruplar or _ONEKLER))


def oznitelik_sutunlari(
    df: pd.DataFrame, gruplar: tuple[str, ...] | None = None
) -> list[str]:
    onekler = _onekler(gruplar)
    return sorted(c for c in df.columns if c.startswith(onekler))


def grup_indeksleri(adlar: list[str], gruplar: tuple[str, ...]) -> list[int]:
    """Ablasyon: seçilen öneklerin (`a`, `z`, `t`) sütun konumları."""
    onekler = _onekler(gruplar)
    idx = [i for i, ad in enumerate(adlar) if ad.startswith(onekler)]
    if not idx:
        raise ValueError(f"gruplar {gruplar} için sütun yok")
    return idx


def yukle(
    strateji: str = "birim_disi",
    parquet: str | Path | None = None,
    gruplar: tuple[str, ...] | None = None,
) -> BolmeVerisi:
    yol = Path(parquet) if parquet else config.ISLENMIS_VERI / "oznitelikler.parquet"
    if not yol.exists():
        raise FileNotFoundError(
            f"{yol} yok — önce: python 02_oznitelik_cikar.py"
        )
    df = pd.read_parquet(yol)
    bol = f"bolme__{strateji}"
    if bol not in df.columns:
        mevcut = [c.removeprefix("bolme__") for c in df.columns if c.startswith("bolme__")]
        raise ValueError(f"'{strateji}' split yok. Mevcut: {mevcut}")

    oz = oznitelik_sutunlari(df, gruplar)
    y = (df["etiket"] == "anomali").astype(int).to_numpy()
    maske = {k: (df[bol] == k).to_numpy() for k in ("train", "val", "test")}

    def meta(k: str) -> pd.DataFrame:
        sutun = _META + [c for c in _META_OPSIYONEL if c in df.columns]
        return df.loc[maske[k], sutun].reset_index(drop=True)

    return BolmeVerisi(
        strateji=strateji,
        oznitelik_adlari=oz,
        X_train=df.loc[maske["train"], oz].to_numpy(np.float64),
        y_train=y[maske["train"]],
        X_val=df.loc[maske["val"], oz].to_numpy(np.float64),
        y_val=y[maske["val"]],
        X_test=df.loc[maske["test"], oz].to_numpy(np.float64),
        y_test=y[maske["test"]],
        meta_test=meta("test"),
        meta_train=meta("train"),
        meta_val=meta("val"),
    )
