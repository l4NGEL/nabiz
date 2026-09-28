"""Tahmini kalan ömre (RUL, çevrim) göre bakım risk seviyesi ve motor sıralaması.

Eşikler bir bakım planlama varsayımıdır (ör. 30 çevrim ≈ bir sonraki planlı
kontrol aralığı), veri ya da gerçek bir filo gereksinimi DEĞİLDİR —
`RISK_ESIKLERI` ile değiştirilebilir. Sıralama metrikleri, bakım ekibinin
"önce hangi motora bakmalı" sorusunda sıralamanın ne kadar işe yaradığını ölçer;
tek tek RUL hatasından (RMSE) farklı bir sorudur.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# (üst sınır dahil, seviye) — küçükten büyüğe; hiçbirine girmeyen NORMAL
RISK_ESIKLERI: tuple[tuple[float, str], ...] = ((30.0, "KRITIK"), (60.0, "YUKSEK"), (90.0, "IZLE"))
SEVIYE_SIRASI = ("KRITIK", "YUKSEK", "IZLE", "NORMAL")


def risk_seviyesi(rul: float, esikler: Sequence[tuple[float, str]] = RISK_ESIKLERI) -> str:
    if not np.isfinite(rul):
        raise ValueError(f"RUL sonlu olmalı: {rul}")
    for sinir, seviye in esikler:
        if rul <= sinir:
            return seviye
    return "NORMAL"


def risk_sirala(birim: Sequence[int], tahmin_rul: Sequence[float]) -> pd.DataFrame:
    """Motorları tahmini RUL'a göre artan (en riskli önce) sırala."""
    df = pd.DataFrame({"birim": np.asarray(birim, dtype=int), "tahmini_rul": np.asarray(tahmin_rul, dtype=float)})
    df["risk"] = [risk_seviyesi(r) for r in df["tahmini_rul"]]
    df = df.sort_values(["tahmini_rul", "birim"], kind="stable").reset_index(drop=True)
    df.insert(0, "sira", np.arange(1, len(df) + 1))
    return df


def siralama_metrikleri(
    gercek_rul: Sequence[float], tahmin_rul: Sequence[float], kritik_sinir: float = 30.0, ust_k: int = 10
) -> dict[str, float]:
    """Bakım önceliklendirme kalitesi.

    - spearman: tahmin ve gerçek RUL sıralamasının uyumu
    - kritik_yakalama: gerçek RUL <= kritik_sinir olan motorların, KRITIK veya
      YUKSEK işaretlenme oranı (kaçırılan kritik motor = en pahalı hata)
    - kritik_kesinlik_ust_k: en riskli ust_k tahminin kaçı gerçekten kritik
    """
    g = np.asarray(gercek_rul, dtype=float)
    t = np.asarray(tahmin_rul, dtype=float)
    if g.shape != t.shape or g.size == 0:
        raise ValueError("gercek_rul ve tahmin_rul aynı uzunlukta ve boş olmamalı")
    kritik = g <= kritik_sinir
    isaretli = np.array([risk_seviyesi(x) in ("KRITIK", "YUKSEK") for x in t])
    ust = np.argsort(t, kind="stable")[: min(ust_k, t.size)]
    return {
        "spearman": float(spearmanr(g, t).statistic),
        "n_kritik": int(kritik.sum()),
        "kritik_yakalama": float(isaretli[kritik].mean()) if kritik.any() else float("nan"),
        "kritik_kesinlik_ust_k": float(kritik[ust].mean()),
        "ust_k": int(ust.size),
    }


def seviye_karisiklik(gercek_rul: Sequence[float], tahmin_rul: Sequence[float]) -> pd.DataFrame:
    """Satır = gerçek seviye, sütun = tahmin seviyesi; motor sayısı."""
    g = [risk_seviyesi(x) for x in gercek_rul]
    t = [risk_seviyesi(x) for x in tahmin_rul]
    tablo = pd.crosstab(pd.Categorical(g, SEVIYE_SIRASI), pd.Categorical(t, SEVIYE_SIRASI), dropna=False)
    tablo.index.name, tablo.columns.name = "gercek", "tahmin"
    return tablo
