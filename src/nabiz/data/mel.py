"""`mel.npz` (log-mel tensörleri) + `oznitelikler.parquet` -> split bazında diziler.

`02_oznitelik_cikar.py --mel` ikisini aynı satır sırasıyla yazar; burada
konuma göre eşleşirler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nabiz import config


@dataclass
class MelBolmeVerisi:
    strateji: str
    sekil: tuple[int, int]                     # (n_mel, T)
    X_train: np.ndarray                        # (n, n_mel, T) float32
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame = field(repr=False)

    @property
    def X_train_normal(self) -> np.ndarray:
        return self.X_train[self.y_train == 0]

    def kirilim(self, sutun: str) -> np.ndarray:
        return self.meta_test[sutun].to_numpy()

    def ozet(self) -> str:
        return (
            f"[{self.strateji}] mel {self.sekil}  train {self.X_train.shape} "
            f"(normal {int((self.y_train == 0).sum())}), "
            f"val {self.X_val.shape}, test {self.X_test.shape} "
            f"(anomali {int(self.y_test.sum())})"
        )


def _yollar(parquet, npz) -> tuple[Path, Path]:
    p = Path(parquet) if parquet else config.ISLENMIS_VERI / "oznitelikler.parquet"
    n = Path(npz) if npz else config.ISLENMIS_VERI / "mel.npz"
    if not n.exists():
        raise FileNotFoundError(f"{n} yok — önce: python 02_oznitelik_cikar.py --mel")
    if not p.exists():
        raise FileNotFoundError(f"{p} yok — önce: python 02_oznitelik_cikar.py --mel")
    return p, n


def yukle(
    strateji: str = "birim_disi",
    parquet: str | Path | None = None,
    npz: str | Path | None = None,
) -> MelBolmeVerisi:
    p_yol, n_yol = _yollar(parquet, npz)
    df = pd.read_parquet(p_yol)
    with np.load(n_yol) as z:
        X = z["X"].astype(np.float32)

    if len(X) != len(df):
        raise ValueError(
            f"mel.npz ({len(X)}) ve parquet ({len(df)}) satır sayısı uyuşmuyor — "
            f"02_oznitelik_cikar.py --mel'i birlikte çalıştır"
        )
    bol = f"bolme__{strateji}"
    if bol not in df.columns:
        mevcut = [c.removeprefix("bolme__") for c in df.columns if c.startswith("bolme__")]
        raise ValueError(f"'{strateji}' split yok. Mevcut: {mevcut}")

    y = (df["etiket"] == "anomali").astype(int).to_numpy()
    m = {k: (df[bol] == k).to_numpy() for k in ("train", "val", "test")}
    return MelBolmeVerisi(
        strateji=strateji,
        sekil=(X.shape[1], X.shape[2]),
        X_train=X[m["train"]], y_train=y[m["train"]],
        X_val=X[m["val"]], y_val=y[m["val"]],
        X_test=X[m["test"]], y_test=y[m["test"]],
        meta_test=df.loc[m["test"], ["makine_tipi", "makine_birimi", "rejim", "ariza_tipi"]].reset_index(drop=True),
    )
