"""Ürün değerlendirmesi: havuz / tip başına / ünite uyarlamalı dedektör.

Eşik, filonun tek F1'i değil. Her makine tipi kendi validasyonundan eşik alır;
yanlış alarmlar toplanıp saate bölünür. Böylece PR-01 ile F1 aynı skorda
karşılaştırılabilir.
"""
from __future__ import annotations

import numpy as np

from nabiz.eval import metrics as MT
from nabiz.models.baseline import Mahalanobis
from nabiz.models.temel import Olcekleyici
from nabiz.models.tip_dedektor import TipDedektoru

KURULUMLAR = ("havuz", "tip", "tip_uyarlama")

_BUTCELER: tuple[tuple[str, str, float], ...] = (
    ("f1", "f1", 0.05),
    ("fpr_1pct", "fpr", 0.01),
    ("fpr_0.1pct", "fpr", 0.001),
    ("pr01", "fpr", 0.0),  # pencere_sn ile doldurulur
)


def _esik_guvenli(y, skor, hedef: str, hedef_fpr: float) -> float | None:
    y = np.asarray(y).astype(int).ravel()
    if y.size == 0 or np.unique(y).size < 2:
        return None
    try:
        return MT.esik_sec(y, skor, hedef=hedef, hedef_fpr=hedef_fpr)
    except ValueError:
        return None


def _skorlar(kurulum, X_train, y_train, tip_train, birim_train, X_val, tip_val, birim_val, X_test, tip_test, birim_test):
    from nabiz.data.uyarlama import BirimOlcek

    normal = np.asarray(y_train).astype(int) == 0
    if int(normal.sum()) < 2:
        raise ValueError("eğitimde en az 2 normal pencere gerekli")
    if kurulum == "tip_uyarlama":
        olcek = BirimOlcek().egit(X_train, birim_train, y_train)
        Xtr = olcek.uygula(X_train[normal], np.asarray(birim_train)[normal], yeniden=False)
        Xva = olcek.uygula(X_val, birim_val, yeniden=True)
        Xte = olcek.uygula(X_test, birim_test, yeniden=True)
    else:
        olc = Olcekleyici().egit(X_train[normal])
        Xtr = olc.uygula(X_train[normal])
        Xva = olc.uygula(X_val)
        Xte = olc.uygula(X_test)

    tipler_n = np.asarray(tip_train)[normal]
    if kurulum == "havuz":
        model = Mahalanobis().egit(Xtr)
        return model.anomali_skoru(Xva), model.anomali_skoru(Xte)
    model = TipDedektoru().egit(Xtr, tipler_n)
    return (
        model.anomali_skoru(Xva, tip_val),
        model.anomali_skoru(Xte, tip_test),
    )


def skorla(
    kurulum, X_train, y_train, tip_train, birim_train,
    X_val, tip_val, birim_val, X_test, tip_test, birim_test,
):
    """Validasyon ve test anomali skorları (yüksek = daha anomalili)."""
    if kurulum not in KURULUMLAR:
        raise ValueError(f"kurulum {kurulum!r} değil; seçenekler: {KURULUMLAR}")
    return _skorlar(
        kurulum, X_train, y_train, tip_train, birim_train,
        X_val, tip_val, birim_val, X_test, tip_test, birim_test,
    )


def _tipli_isletme(y_te, s_te, tip_te, y_va, s_va, tip_va, hedef, hedef_fpr, saat) -> dict:
    tp = fp = tn = fn = 0
    for tip in np.unique(tip_te):
        mte = tip_te == tip
        mva = tip_va == tip
        esik = _esik_guvenli(y_va[mva], s_va[mva], hedef, hedef_fpr) if mva.any() else None
        if esik is None:
            esik = _esik_guvenli(y_va, s_va, hedef, hedef_fpr)
        if esik is None:
            continue
        met = MT.esikte_metrikler(y_te[mte], s_te[mte], esik)
        tp += met["tp"]
        fp += met["fp"]
        tn += met["tn"]
        fn += met["fn"]
    if tp + fp + tn + fn == 0:
        raise ValueError("işletme noktası seçilemedi")
    return {
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "yanlis_alarm_saat": (fp / saat) if saat > 0 else float("nan"),
        "fp": fp,
        "tp": tp,
    }


def degerlendir_diziler(
    X_train, y_train, tip_train, birim_train,
    X_val, y_val, tip_val, birim_val,
    X_test, y_test, tip_test, birim_test,
    kurulum: str = "tip_uyarlama",
    pencere_sn: float = 2.0,
) -> dict:
    """Tek kurulum. Dönen sözlük makro AUC ve tip-başına işletme noktalarını taşır."""
    if kurulum not in KURULUMLAR:
        raise ValueError(f"kurulum {kurulum!r} değil; seçenekler: {KURULUMLAR}")
    y_test = np.asarray(y_test).astype(int).ravel()
    y_val = np.asarray(y_val).astype(int).ravel()
    tip_test = np.asarray(tip_test).ravel()
    tip_val = np.asarray(tip_val).ravel()
    s_va, s_te = _skorlar(
        kurulum, X_train, y_train, tip_train, birim_train,
        X_val, tip_val, birim_val, X_test, tip_test, birim_test,
    )

    tipler: dict[str, dict] = {}
    auc_list: list[float] = []
    pauc_list: list[float] = []
    for tip in np.unique(tip_test):
        m = tip_test == tip
        if np.unique(y_test[m]).size < 2:
            continue
        a = MT.auc(y_test[m], s_te[m])
        p = MT.pauc(y_test[m], s_te[m])
        tipler[str(tip)] = {
            "auc": a,
            "pauc": p,
            "n": int(m.sum()),
            "n_anomali": int(y_test[m].sum()),
        }
        auc_list.append(a)
        pauc_list.append(p)
    if not auc_list:
        raise ValueError("testte iki sınıflı makine tipi yok")

    saat = float((y_test == 0).sum()) * pencere_sn / 3600.0
    butceler = []
    for ad, hedef, fpr in _BUTCELER:
        if ad == "pr01":
            fpr = MT.fpr_butcesi(MT.PR01_ALARM_SAAT, pencere_sn)
        butceler.append((ad, hedef, fpr))

    isletme = {}
    for ad, hedef, fpr in butceler:
        try:
            isletme[ad] = _tipli_isletme(
                y_test, s_te, tip_test, y_val, s_va, tip_val, hedef, fpr, saat
            )
        except ValueError:
            continue

    return {
        "kurulum": kurulum,
        "havuz_auc": MT.auc(y_test, s_te),
        "havuz_pauc": MT.pauc(y_test, s_te),
        "makro_auc": float(np.mean(auc_list)),
        "makro_pauc": float(np.mean(pauc_list)),
        "tipler": tipler,
        "isletme": isletme,
        "saat": saat,
    }
