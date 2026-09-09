"""Anomali modeli protokolü + ortak yardımcılar."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomaliModeli(Protocol):
    """Gözetimsiz anomali dedektörü arayüzü."""

    ad: str

    def egit(self, X: np.ndarray) -> AnomaliModeli: ...

    def anomali_skoru(self, X: np.ndarray) -> np.ndarray:
        """(n,) skor dizisi — yüksek = daha anomalili."""
        ...


class Olcekleyici:
    """z-skor standardizasyonu (train istatistikleriyle). sklearn'siz, taşınabilir."""

    def __init__(self) -> None:
        self.ort_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def egit(self, X: np.ndarray) -> Olcekleyici:
        X = np.asarray(X, dtype=np.float64)
        self.ort_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def uygula(self, X: np.ndarray) -> np.ndarray:
        if self.ort_ is None:
            raise RuntimeError("Olcekleyici.egit() çağrılmadı")
        return (np.asarray(X, dtype=np.float64) - self.ort_) / self.std_

    def egit_uygula(self, X: np.ndarray) -> np.ndarray:
        return self.egit(X).uygula(X)


def dogrula_2b(X: np.ndarray, ad: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{ad} 2B (n_ornek, n_oznitelik) olmalı, shape={X.shape}")
    if X.size == 0:
        raise ValueError(f"{ad} boş")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{ad} sonlu değil (NaN/inf)")
    return X
