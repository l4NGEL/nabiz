"""Anomali / sınıflandırma / RUL metrikleri.

Konvansiyon: `y` içinde 1 = anomali, 0 = normal. `skor` yüksek = daha anomalili.
pAUC DCASE Task 2 ile aynı (sklearn `max_fpr`, McClish düzeltmeli).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _ikili(y, skor) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(skor, dtype=np.float64).ravel()
    if y.size != s.size:
        raise ValueError(f"y ({y.size}) ve skor ({s.size}) uzunluğu farklı")
    if y.size == 0:
        raise ValueError("boş girdi")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"y yalnız 0/1 olmalı, görülen: {np.unique(y)}")
    return y, s


def auc(y, skor) -> float:
    y, s = _ikili(y, skor)
    if y.min() == y.max():
        raise ValueError("AUC için her iki sınıf da gerekli")
    return float(roc_auc_score(y, s))


def pauc(y, skor, maks_fpr: float = 0.1) -> float:
    """Kısmi AUC — FPR ∈ [0, maks_fpr] bölgesi (DCASE Task 2, p=0.1)."""
    y, s = _ikili(y, skor)
    if y.min() == y.max():
        raise ValueError("pAUC için her iki sınıf da gerekli")
    return float(roc_auc_score(y, s, max_fpr=maks_fpr))


def esik_sec(y_val, skor_val, hedef: str = "f1", hedef_fpr: float = 0.05) -> float:
    """Validasyon üzerinde eşik seç.

    hedef='f1'  → F1'i maksimize eden eşik
    hedef='fpr' → FPR ≈ hedef_fpr veren eşik (operasyonel yanlış alarm bütçesi)
    """
    y, s = _ikili(y_val, skor_val)
    adaylar = np.unique(s)
    if adaylar.size > 512:
        adaylar = np.quantile(s, np.linspace(0, 1, 512))
    if hedef == "fpr":
        fpr, _tpr, esikler = roc_curve(y, s)
        i = int(np.argmin(np.abs(fpr - hedef_fpr)))
        return float(esikler[i])
    en_iyi_f1, en_iyi_esik = -1.0, float(adaylar[0])
    for e in adaylar:
        tahmin = s >= e
        tp = int(np.sum(tahmin & (y == 1)))
        fp = int(np.sum(tahmin & (y == 0)))
        fn = int(np.sum(~tahmin & (y == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > en_iyi_f1:
            en_iyi_f1, en_iyi_esik = f1, float(e)
    return en_iyi_esik


def esikte_metrikler(y, skor, esik: float) -> dict[str, float]:
    y, s = _ikili(y, skor)
    tahmin = s >= esik
    tp = int(np.sum(tahmin & (y == 1)))
    fp = int(np.sum(tahmin & (y == 0)))
    tn = int(np.sum(~tahmin & (y == 0)))
    fn = int(np.sum(~tahmin & (y == 1)))
    return {
        "esik": float(esik),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def fpr_butcesi(alarm_saat_basi: float, pencere_sn: float) -> float:
    """Hedef yanlış alarm/saat → pencere başına FPR.

    Saatlik alarm = FPR × (3600 / pencere_sn). PR-01 (≤ 1 alarm / 24 saat,
    2 sn pencere) ≈ 2.3e-5 FPR.
    """
    if alarm_saat_basi < 0 or pencere_sn <= 0:
        raise ValueError("alarm_saat_basi ≥ 0 ve pencere_sn > 0 olmalı")
    return float(alarm_saat_basi * pencere_sn / 3600.0)


# PR-01: nominal çalışmada ≤ 1 yanlış alarm / 24 saat
PR01_ALARM_SAAT = 1.0 / 24.0


def yanlis_alarm_saat(y, skor, esik: float, toplam_saat: float) -> float:
    """Normal pencerelerde pozitif işaret sayısı / saat (PR-01 vekili)."""
    y, s = _ikili(y, skor)
    if toplam_saat <= 0:
        raise ValueError("toplam_saat > 0 olmalı")
    fp = int(np.sum((s >= esik) & (y == 0)))
    return fp / toplam_saat


def ece(y_true, olasilik, n_bin: int = 15) -> float:
    """Expected Calibration Error (Guo ve ark.) — olasılık `olasilik` = P(sınıf 1).

    Her bin için: |doğruluk - güven|, bin boyutuyla ağırlıklı. Güven = tahmin
    edilen sınıfın olasılığı (max(p, 1-p)).
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(olasilik, dtype=np.float64).ravel()
    if y.size != p.size or y.size == 0:
        raise ValueError("y_true ve olasilik aynı uzunlukta ve boş olmamalı")
    tahmin = (p >= 0.5).astype(int)
    guven = np.where(tahmin == 1, p, 1.0 - p)
    dogru = (tahmin == y).astype(float)

    kenar = np.linspace(0.0, 1.0, n_bin + 1)
    e = 0.0
    for lo, hi in zip(kenar[:-1], kenar[1:], strict=True):
        m = (guven > lo) & (guven <= hi) if lo > 0 else (guven >= lo) & (guven <= hi)
        if not m.any():
            continue
        e += m.mean() * abs(dogru[m].mean() - guven[m].mean())
    return float(e)


def cmapss_skoru(gercek_rul, tahmin_rul) -> float:
    """C-MAPSS asimetrik RUL skoru — geç tahmin (d>0) daha ağır cezalı. Düşük iyi."""
    g = np.asarray(gercek_rul, dtype=np.float64).ravel()
    t = np.asarray(tahmin_rul, dtype=np.float64).ravel()
    if g.size != t.size or g.size == 0:
        raise ValueError("gercek_rul ve tahmin_rul aynı uzunlukta ve boş olmamalı")
    d = t - g
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def bootstrap_ci(
    metrik_fn, y, skor, n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(ortalama, alt, üst) — metrik_fn(y, skor) için bootstrap güven aralığı."""
    y, s = _ikili(y, skor)
    rng = np.random.default_rng(seed)
    degerler: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, y.size, y.size)
        try:
            degerler.append(float(metrik_fn(y[idx], s[idx])))
        except ValueError:
            continue  # tek sınıflı yeniden örnekleme
    if not degerler:
        raise ValueError("bootstrap: hiçbir örnek iki sınıf içermedi")
    arr = np.asarray(degerler)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def anomali_raporu(
    y_test,
    skor_test,
    y_val=None,
    skor_val=None,
    kirilim: dict[str, np.ndarray] | None = None,
    toplam_saat: float | None = None,
) -> dict:
    """Tam anomali değerlendirme sözlüğü: AUC, pAUC (+CI), eşik, eşikte metrikler,
    opsiyonel kırılım (makine tipi / rejim / arıza tipi) ve yanlış alarm/saat."""
    y_test, s_test = _ikili(y_test, skor_test)
    rapor: dict = {
        "n": int(y_test.size),
        "n_anomali": int(y_test.sum()),
        "auc": auc(y_test, s_test),
        "pauc": pauc(y_test, s_test),
    }
    rapor["auc_ort"], rapor["auc_alt"], rapor["auc_ust"] = bootstrap_ci(auc, y_test, s_test, n=500)

    if y_val is not None and skor_val is not None:
        esik = esik_sec(y_val, skor_val, hedef="f1")
        rapor["esik_kaynak"] = "val"
    else:
        esik = esik_sec(y_test, s_test, hedef="f1")
        rapor["esik_kaynak"] = "test (sızıntı — yalnız gösterim)"
    rapor["esikte"] = esikte_metrikler(y_test, s_test, esik)

    if toplam_saat:
        rapor["yanlis_alarm_saat"] = yanlis_alarm_saat(y_test, s_test, esik, toplam_saat)

    if kirilim:
        rapor["kirilim"] = {}
        for ad, deger in kirilim.items():
            deger = np.asarray(deger).ravel()
            alt = {}
            for grup in np.unique(deger):
                m = deger == grup
                if y_test[m].min() != y_test[m].max():
                    alt[str(grup)] = {
                        "n": int(m.sum()),
                        "auc": auc(y_test[m], s_test[m]),
                        "pauc": pauc(y_test[m], s_test[m]),
                    }
            rapor["kirilim"][ad] = alt
    return rapor
