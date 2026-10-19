"""Makine tipi başına ayrı anomali modeli.

Fan ve pompa aynı kovaryansa sığdırılınca skor ölçekleri karışır: tip içi
ayrım, havuzlanmış AUC'den iyi olsa da tek eşik anlamsızlaşır. Her tip kendi
normaliyle eğitilir; eğitimde görülmeyen tip filo yedeğine düşer.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nabiz.models.baseline import Mahalanobis
from nabiz.models.temel import dogrula_2b


class TipDedektoru:
    """`fabrika()` her tip için yeni bir dedektör döndürür (öntanım: Mahalanobis)."""

    def __init__(self, fabrika: Callable[[], Mahalanobis] | None = None) -> None:
        self._fabrika = fabrika or Mahalanobis
        self.ad = f"tip_{getattr(self._fabrika(), 'ad', 'model')}"
        self._modeller: dict[str, Mahalanobis] = {}
        self._yedek: Mahalanobis | None = None

    def egit(self, X: np.ndarray, tipler: np.ndarray) -> TipDedektoru:
        X = dogrula_2b(X)
        tipler = np.asarray(tipler).ravel()
        if tipler.size != X.shape[0]:
            raise ValueError("tipler ve X aynı uzunlukta olmalı")
        self._modeller = {}
        for tip in np.unique(tipler):
            m = tipler == tip
            if int(m.sum()) < 2:
                continue
            self._modeller[str(tip)] = self._fabrika().egit(X[m])
        self._yedek = self._fabrika().egit(X)
        return self

    def anomali_skoru(self, X: np.ndarray, tipler: np.ndarray) -> np.ndarray:
        if self._yedek is None:
            raise RuntimeError("TipDedektoru.egit() çağrılmadı")
        X = dogrula_2b(X)
        tipler = np.asarray(tipler).ravel()
        if tipler.size != X.shape[0]:
            raise ValueError("tipler ve X aynı uzunlukta olmalı")
        skor = np.empty(X.shape[0], dtype=np.float64)
        for tip in np.unique(tipler):
            m = tipler == tip
            model = self._modeller.get(str(tip), self._yedek)
            skor[m] = model.anomali_skoru(X[m])
        return skor
