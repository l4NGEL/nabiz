"""Yenilmesi gereken baseline anomali dedektörleri.

Derin modeller (autoencoder, CRNN, ...) bu üçünü geçmeli — geçmezse
docs/test-plani.md §2 gereği dürüstçe raporlanır.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest

from nabiz.models.temel import dogrula_2b


class RMSEsigi:
    """En basit dedektör: pencere RMS'i (`a_rms` özniteliği) doğrudan skor.

    Eğitimde bir şey öğrenmez; `a_rms` sütun indeksini alır. Referans taban.
    """

    ad = "rms_esigi"

    def __init__(self, oznitelik_adlari: list[str], oznitelik: str = "a_rms") -> None:
        if oznitelik not in oznitelik_adlari:
            raise ValueError(f"'{oznitelik}' öznitelik listesinde yok")
        self._i = oznitelik_adlari.index(oznitelik)

    def egit(self, X: np.ndarray) -> RMSEsigi:
        dogrula_2b(X)
        return self

    def anomali_skoru(self, X: np.ndarray) -> np.ndarray:
        return dogrula_2b(X)[:, self._i]


class IzolasyonOrmani:
    """IsolationForest — ağaç tabanlı, ölçek-duyarsız."""

    ad = "izolasyon_ormani"

    def __init__(self, n_agac: int = 200, seed: int = 0) -> None:
        self._m = IsolationForest(
            n_estimators=n_agac, random_state=seed, n_jobs=1, contamination="auto"
        )

    def egit(self, X: np.ndarray) -> IzolasyonOrmani:
        self._m.fit(dogrula_2b(X))
        return self

    def anomali_skoru(self, X: np.ndarray) -> np.ndarray:
        # score_samples: yüksek = normal → negatifle → yüksek = anomali
        return -self._m.score_samples(dogrula_2b(X))


class Mahalanobis:
    """Normal verinin Gaussian'ı; skor = Mahalanobis mesafesi (Ledoit-Wolf shrink)."""

    ad = "mahalanobis"

    def __init__(self) -> None:
        self._cov = EmpiricalCovariance(assume_centered=False)
        self._mu: np.ndarray | None = None

    def egit(self, X: np.ndarray) -> Mahalanobis:
        X = dogrula_2b(X)
        self._cov.fit(X)
        self._mu = X.mean(axis=0)
        return self

    def anomali_skoru(self, X: np.ndarray) -> np.ndarray:
        X = dogrula_2b(X)
        return np.sqrt(np.clip(self._cov.mahalanobis(X), 0.0, None))
