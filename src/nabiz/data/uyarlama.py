"""Ünite bazlı ölçek — görülmemiş makinenin kendi taban çizgisine oturtma.

Eğitimde her birimin normal pencerelerinden medyan ve MAD alınır. Hedef
bölmede (val/test) birim eğitimde yoksa ya da `yeniden=True` ise istatistik
etiketsiz, o partinin kendi satırlarından hesaplanır. Aykırı değerlerin
merkezi kaydırmaması için ortalama yerine medyan kullanılır.

Bu, etiket kullanmaz: sahada makinenin kendi geçmişi vardır, eğitim
filodaki başka bir ünitenin "normal"i yoktur.
"""
from __future__ import annotations

import numpy as np

_MAD_OLCEK = 1.4826
_MIN_ORNEK = 8
_EPS = 1e-9


def _merkez_olcek(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ort = np.median(X, axis=0)
    mad = np.median(np.abs(X - ort), axis=0) * _MAD_OLCEK
    mad[mad < _EPS] = 1.0
    return ort, mad


class BirimOlcek:
    """Birim kimliğine göre sağlam (medyan/MAD) standardizasyon."""

    def __init__(self) -> None:
        self.istatistik: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.yedek_ort: np.ndarray | None = None
        self.yedek_std: np.ndarray | None = None

    def egit(self, X: np.ndarray, birimler: np.ndarray, y: np.ndarray) -> BirimOlcek:
        X = np.asarray(X, dtype=np.float64)
        birimler = np.asarray(birimler).ravel()
        y = np.asarray(y).astype(int).ravel()
        if X.shape[0] != birimler.size or X.shape[0] != y.size:
            raise ValueError("X, birimler ve y aynı uzunlukta olmalı")
        normal = y == 0
        if int(normal.sum()) < _MIN_ORNEK:
            raise ValueError("eğitimde yeterince normal pencere yok")
        self.yedek_ort, self.yedek_std = _merkez_olcek(X[normal])
        self.istatistik = {}
        for birim in np.unique(birimler[normal]):
            m = normal & (birimler == birim)
            if int(m.sum()) < _MIN_ORNEK:
                continue
            self.istatistik[str(birim)] = _merkez_olcek(X[m])
        return self

    def uygula(
        self, X: np.ndarray, birimler: np.ndarray, yeniden: bool = False
    ) -> np.ndarray:
        """`yeniden=False`: görülmüş birimde eğitim istatistiği.

        `yeniden=True`: her birim, bu partinin kendi satırlarıyla ölçeklenir
        (yeni rejim / yeni ünite). Az örnekte yedek (filo) istatistiğine düşer.
        """
        if self.yedek_ort is None or self.yedek_std is None:
            raise RuntimeError("BirimOlcek.egit() çağrılmadı")
        X = np.asarray(X, dtype=np.float64)
        birimler = np.asarray(birimler).ravel()
        if X.shape[0] != birimler.size:
            raise ValueError("X ve birimler aynı uzunlukta olmalı")
        out = np.empty_like(X, dtype=np.float64)
        for birim in np.unique(birimler):
            m = birimler == birim
            anahtar = str(birim)
            if yeniden or anahtar not in self.istatistik:
                if int(m.sum()) >= _MIN_ORNEK:
                    ort, std = _merkez_olcek(X[m])
                else:
                    ort, std = self.yedek_ort, self.yedek_std
            else:
                ort, std = self.istatistik[anahtar]
            out[m] = (X[m] - ort) / std
        return out
