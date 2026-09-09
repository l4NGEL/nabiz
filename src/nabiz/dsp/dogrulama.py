"""Ortak giriş doğrulama yardımcıları — DSP modülleri paylaşır."""
from __future__ import annotations

import numpy as np


def sonlu_1b(x, ad: str = "sinyal") -> np.ndarray:
    """1B, boş olmayan, sonlu float64 diziye çevir; aksi halde ValueError."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{ad} 1B olmalı, shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{ad} boş")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{ad} sonlu değil (NaN/inf)")
    return x


def pozitif(deger: float, ad: str) -> float:
    if not np.isfinite(deger) or deger <= 0:
        raise ValueError(f"{ad} > 0 olmalı, verilen: {deger}")
    return float(deger)
