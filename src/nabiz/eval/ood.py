"""OOD (dağılım-dışı) sağlamlık kontrolü.

Bir anomali modeli yalnız normal veriyle eğitilir; "anomali" ayrımını
görülmüş arıza örüntülerine göre öğrenir. İyi bir dedektör hiç görmediği
girdiyi de (yabancı makine tipi, saf gürültü, öznitelikler-arası ilişkisi
bozulmuş veri) normalden ayırt etmeli — aksi halde "kendinden emin yanlış"
(sessizce normal der). Bu modül OOD ayrımını mevcut AUC makinesiyle ölçer:
OOD = pozitif sınıf, iç-dağılım normal = negatif.
"""
from __future__ import annotations

import numpy as np

from nabiz.eval.metrics import auc as _auc


def gurultu_uret(X_referans: np.ndarray, seed: int = 0) -> np.ndarray:
    """Referansla aynı şekil/ölçekte saf Gauss gürültüsü (öznitelik korelasyonu yok)."""
    X_referans = np.asarray(X_referans)
    rng = np.random.default_rng(seed)
    ort = X_referans.reshape(X_referans.shape[0], -1).mean(axis=0)
    std = X_referans.reshape(X_referans.shape[0], -1).std(axis=0)
    duz = rng.normal(ort, std + 1e-9, size=(X_referans.shape[0], ort.size))
    return duz.reshape(X_referans.shape).astype(X_referans.dtype)


def karistir_uret(X_referans: np.ndarray, seed: int = 0) -> np.ndarray:
    """Her öznitelik sütununu örnekler arasında bağımsız karıştır — marjinal
    dağılımlar aynı kalır, öznitelikler-arası korelasyon (yapı) bozulur."""
    X_referans = np.asarray(X_referans)
    rng = np.random.default_rng(seed)
    duz = X_referans.reshape(X_referans.shape[0], -1).copy()
    for j in range(duz.shape[1]):
        rng.shuffle(duz[:, j])
    return duz.reshape(X_referans.shape)


def ood_karsilastir(skor_ic_dagilim: np.ndarray, skor_ood: np.ndarray) -> dict:
    """OOD'u iç-dağılım normalden ayırma gücü + skor özeti.

    `ayrim_auc` yüksek (→1) = model OOD'u doğru şekilde daha anomalili buluyor.
    ~0.5 = OOD ile normali ayıramıyor (kendinden emin yanlış riski).
    """
    s_ic = np.asarray(skor_ic_dagilim, dtype=np.float64).ravel()
    s_ood = np.asarray(skor_ood, dtype=np.float64).ravel()
    if s_ic.size == 0 or s_ood.size == 0:
        raise ValueError("skor dizileri boş olamaz")
    y = np.r_[np.zeros(s_ic.size), np.ones(s_ood.size)].astype(int)
    s = np.r_[s_ic, s_ood]
    return {
        "ic_dagilim_ort": float(s_ic.mean()),
        "ood_ort": float(s_ood.mean()),
        "ood_ic_dagilimdan_yuksek_oran": float((s_ood > np.median(s_ic)).mean()),
        "ayrim_auc": _auc(y, s),
    }
