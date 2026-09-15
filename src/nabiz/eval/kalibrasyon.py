"""Anomali skorunu kalibre edilmiş olasılığa çevirme — Platt scaling.

AE/baseline skorları (yeniden yapılandırma MSE, mesafe, ...) doğal bir
olasılık değildir — büyüklükleri modelden modele, veri setinden veri setine
değişir. Küçük, etiketli bir validasyon kümesiyle 1B lojistik regresyon
(Platt scaling) uydurulup P(anomali) elde edilir; kalite `eval/metrics.py::ece`
ile ölçülür. Eğitim (AE) hâlâ gözetimsizdir — yalnız kalibrasyon/eşik
etiket kullanır (zaten `esik_sec` için de aynı prensip).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattKalibratoru:
    """Ham anomali skorunu [0,1] kalibre olasılığa çevirir (1B lojistik regresyon)."""

    def __init__(self, C: float = 1.0) -> None:
        self._C = C
        self._lr: LogisticRegression | None = None
        self._ort: float | None = None
        self._std: float | None = None

    def egit(self, skor_val: np.ndarray, y_val: np.ndarray) -> PlattKalibratoru:
        s = np.asarray(skor_val, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y_val).astype(int).ravel()
        if s.shape[0] != y.shape[0]:
            raise ValueError(f"skor_val ({s.shape[0]}) ve y_val ({y.shape[0]}) uzunluğu farklı")
        if len(np.unique(y)) < 2:
            raise ValueError("kalibrasyon için validasyonda iki sınıf da (normal+anomali) gerekli")
        self._ort, self._std = float(s.mean()), float(s.std()) or 1.0
        self._lr = LogisticRegression(C=self._C)
        self._lr.fit((s - self._ort) / self._std, y)
        return self

    def olasilik(self, skor: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("PlattKalibratoru.egit() çağrılmadı")
        s = np.asarray(skor, dtype=np.float64).reshape(-1, 1)
        return self._lr.predict_proba((s - self._ort) / self._std)[:, 1]
