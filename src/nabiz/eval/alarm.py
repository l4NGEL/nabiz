"""Dosya düzeyinde alarm — PR-01 işletme noktası.

MIMII kaydı ~10 sn sürer; 2 sn pencereler örtüşür. Arıza kaydın tamamına
yayılır, tek pencere gürültüsü yayılmaz. Karar birimi penceredir değil,
dosyadır: skor = pencerelerin medyanı (veya p90). Bir yanlış alarm = bir
normal dosyanın işaretlenmesi.

Bütçe saatte olay sayısınadır. Eldeki normal kayıt 24 saatten kısaysa
≤ 1 alarm / 24 saat, o kayıtta sıfır yanlış alarm demektir — eşik,
validasyon normallerinin tümünün üstüne konur.
"""
from __future__ import annotations

import numpy as np

from nabiz.eval import metrics as MT

YONTEMLER = ("medyan", "p90")
SUREKLILIK_K = (1, 3, 6)  # ~10 sn, ~30 sn, ~60 sn
Z_IZGARA = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
Z_MARJ = 0.5  # eşik yarım sigma gevşese de validasyon bütçesi durmalı


def _ozet(skor: np.ndarray, yontem: str) -> float:
    if yontem == "medyan":
        return float(np.median(skor))
    if yontem == "p90":
        return float(np.quantile(skor, 0.9))
    if yontem == "tepe3":
        k = min(3, skor.size)
        return float(np.mean(np.partition(skor, -k)[-k:]))
    raise ValueError(f"yontem {yontem!r} değil; seçenekler: medyan, p90, tepe3")


def dosya_sure_sn(pencere_idx: np.ndarray, pencere_sn: float, hop_sn: float) -> float:
    """Örtüşen pencerelerin kapladığı süre."""
    if pencere_idx.size == 0:
        raise ValueError("pencere_idx boş")
    if hop_sn < 0 or pencere_sn <= 0:
        raise ValueError("pencere_sn > 0 ve hop_sn ≥ 0 olmalı")
    return float((pencere_idx.max() - pencere_idx.min()) * hop_sn + pencere_sn)


def dosyalara_indir(
    skor, y, dosya, tip, pencere_idx, yontem: str, pencere_sn: float, hop_sn: float,
    birim=None,
) -> dict[str, np.ndarray]:
    """Pencere skorunu dosya başına tek skora indirger."""
    skor = np.asarray(skor, dtype=np.float64).ravel()
    y = np.asarray(y).astype(int).ravel()
    dosya = np.asarray(dosya).ravel()
    tip = np.asarray(tip).ravel()
    pencere_idx = np.asarray(pencere_idx).ravel()
    n = skor.size
    if birim is None:
        birim = np.array(["_"] * n)
    else:
        birim = np.asarray(birim).ravel()
    if not (y.size == dosya.size == tip.size == pencere_idx.size == birim.size == n):
        raise ValueError("skor, y, dosya, tip, pencere_idx, birim aynı uzunlukta olmalı")
    if n == 0:
        raise ValueError("boş girdi")

    _, ilk = np.unique(dosya, return_index=True)
    kimlikler = dosya[np.sort(ilk)]
    s_d = np.empty(kimlikler.size, dtype=np.float64)
    y_d = np.empty(kimlikler.size, dtype=int)
    tip_d = np.empty(kimlikler.size, dtype=tip.dtype)
    birim_d = np.empty(kimlikler.size, dtype=birim.dtype)
    sure = np.empty(kimlikler.size, dtype=np.float64)
    for i, kim in enumerate(kimlikler):
        m = dosya == kim
        s_d[i] = _ozet(skor[m], yontem)
        y_d[i] = int(y[m].max())
        tip_d[i] = tip[m][0]
        birim_d[i] = birim[m][0]
        sure[i] = dosya_sure_sn(pencere_idx[m], pencere_sn, hop_sn)
    return {
        "skor": s_d,
        "y": y_d,
        "tip": tip_d,
        "sure_sn": sure,
        "dosya": kimlikler,
        "birim": birim_d,
        "oturum": np.array([_oturum_adi(k) for k in kimlikler]),
    }


def _birim_z(paket: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Dosya skorunu birimin kendi medyan/MAD'ine böl.

    Eşik ham Mahalanobis'te üniteden üniteye taşınmıyor. Referans etiketsizdir
    ve birimin tüm dosyalarından gelir; arıza oranı %50'nin altındaysa medyan
    normalde kalır.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in paket.items()}
    skor = out["skor"].astype(np.float64, copy=True)
    birim = out["birim"]
    z = np.empty_like(skor)
    for b in np.unique(birim):
        m = birim == b
        s = skor[m]
        med = float(np.median(s))
        sapma = float(np.median(np.abs(s - med)) * 1.4826)
        if sapma < 1e-9:
            sapma = 1.0
        z[m] = (s - med) / sapma
    out["skor"] = z
    return out


def _oturum_adi(dosya) -> str:
    """Kayıt klasörü. İki SNR oturumu yol sıralamasıyla birbirine dikilmez."""
    s = str(dosya).replace("\\", "/")
    return s.rsplit("/", 1)[0] if "/" in s else s


def olay_sayisi(sicak: np.ndarray, k: int) -> int:
    """Eşikteki ardışık dosyalardan kaç ayrı alarm olayı çıkar.

    Tek sıcak dosya, k>1 ise olay sayılmaz. Olay, k'ıncı sıcak dosyada bir kez
    açılır; soğuk dosya görününce kapanır.
    """
    if k < 1:
        raise ValueError("k ≥ 1 olmalı")
    sicak = np.asarray(sicak, dtype=bool).ravel()
    olay = 0
    ardisik = 0
    acik = False
    for h in sicak:
        if h:
            ardisik += 1
            if ardisik >= k and not acik:
                olay += 1
                acik = True
        else:
            ardisik = 0
            acik = False
    return olay


def _oturum_skorlari(paket: dict[str, np.ndarray], normal: bool) -> list[np.ndarray]:
    y = paket["y"]
    maske = (y == 0) if normal else (y == 1)
    if not np.any(maske):
        return []
    skor = paket["skor"]
    dosya = paket["dosya"]
    oturum = paket["oturum"]
    diziler = []
    for ot in np.unique(oturum[maske]):
        m = np.flatnonzero(maske & (oturum == ot))
        m = m[np.argsort(dosya[m], kind="mergesort")]
        diziler.append(skor[m])
    return diziler


def _olay_toplam(diziler: list[np.ndarray], esik: float, k: int) -> int:
    return sum(olay_sayisi(s >= esik, k) for s in diziler)


def isletme_sec(
    paket_val: dict[str, np.ndarray],
    alarm_saat_basi: float,
    k_aday: tuple[int, ...] = SUREKLILIK_K,
    z_aday: tuple[float, ...] = Z_IZGARA,
) -> dict:
    """z ve k yalnız validasyondan. Test etiketine bakmaz."""
    if alarm_saat_basi < 0:
        raise ValueError("alarm_saat_basi ≥ 0 olmalı")
    paket_val = _birim_z(paket_val)
    secenek: list[dict] = []
    for z in z_aday:
        for k in k_aday:
            val_olc = _olc(paket_val, z, k, alarm_saat_basi)
            marj = _olc(paket_val, z - Z_MARJ, k, alarm_saat_basi)
            izin = int(np.floor(alarm_saat_basi * val_olc["normal_saat"] + 1e-9))
            secenek.append({
                "k": int(k),
                "esik": float(z),
                "val_recall": val_olc["recall"],
                "val_olay": val_olc["olay_fp"],
                "val_izin": izin,
                "uygun": val_olc["olay_fp"] <= izin and marj["olay_fp"] <= izin,
            })
    uygun = [s for s in secenek if s["uygun"]]
    havuz = uygun or secenek
    en_iyi = max(s["val_recall"] for s in havuz)
    yakin = [s for s in havuz if s["val_recall"] >= en_iyi - 1e-9]
    # Dosya recall eşiğe bağlıdır, k'ya değil. Aynı recall'da en uzun
    # süreklilik yanlış olay sayısını düşürür; eşiği gevşetmez.
    return min(yakin, key=lambda s: (-s["k"], -s["val_recall"]))


def butce_sureklilik(
    paket_val: dict[str, np.ndarray],
    paket_test: dict[str, np.ndarray],
    alarm_saat_basi: float,
    k_aday: tuple[int, ...] = SUREKLILIK_K,
    z_aday: tuple[float, ...] = Z_IZGARA,
) -> dict:
    """Sabit z ızgarası ve süreklilik. Seçim yalnız validasyona bakar.

    En düşük eşiği aramak, validasyon ünitesinin kuyruğuna yapışır ve
    görülmemiş ünitede patlar. Adaylar sabit z değerleridir; eşik yarım
    sigma gevşetilince de bütçe duruyorsa aday geçerlidir.
    """
    sec = isletme_sec(paket_val, alarm_saat_basi, k_aday, z_aday)
    olcum = _olc(_birim_z(paket_test), sec["esik"], sec["k"], alarm_saat_basi)
    return {**sec, **olcum, "secildi_k": True}


def alarm_olaylari(paket: dict[str, np.ndarray], esik: float, k: int) -> list[dict]:
    """Etiketsiz. Oturum içinde k ardışık sıcak dosyada bir olay açılır."""
    if k < 1:
        raise ValueError("k ≥ 1 olmalı")
    oturum = paket["oturum"]
    dosya = paket["dosya"]
    skor = paket["skor"]
    birim = paket["birim"]
    olaylar: list[dict] = []
    for ot in np.unique(oturum):
        m = np.flatnonzero(oturum == ot)
        m = m[np.argsort(dosya[m], kind="mergesort")]
        ardisik = 0
        acik = False
        for i in m:
            if skor[i] >= esik:
                ardisik += 1
                if ardisik >= k and not acik:
                    acik = True
                    olaylar.append({
                        "dosya": str(dosya[i]),
                        "birim": str(birim[i]),
                        "oturum": str(ot),
                        "skor": float(skor[i]),
                    })
            else:
                ardisik = 0
                acik = False
    return olaylar


def _olc(paket: dict[str, np.ndarray], esik: float, k: int, alarm_saat_basi: float) -> dict:
    normal = paket["y"] == 0
    saat = float(paket["sure_sn"][normal].sum()) / 3600.0
    olay = _olay_toplam(_oturum_skorlari(paket, normal=True), esik, k)
    anom = paket["skor"][paket["y"] == 1]
    recall = float(np.mean(anom >= esik)) if anom.size else 0.0
    oran = (olay / saat) if saat > 0 else float("nan")
    return {
        "recall": recall,
        "olay_fp": int(olay),
        "normal_saat": saat,
        "yanlis_alarm_saat": oran,
        "pr01": bool(saat > 0 and oran <= alarm_saat_basi + 1e-9),
    }


def esik_fp_butcesi(y, skor, toplam_saat: float, alarm_saat_basi: float) -> float:
    """Yanlış alarm/saat ≤ bütçe iken en çok tespiti bırakan eşik.

    İzin verilen yanlış alarm sayısı = floor(bütçe × saat). Bu 1'den küçükse
    eşik, en yüksek normal skorun hemen üstüdür (sıfır yanlış alarm).
    """
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(skor, dtype=np.float64).ravel()
    if y.size != s.size or y.size == 0:
        raise ValueError("y ve skor aynı uzunlukta ve boş olmamalı")
    if toplam_saat <= 0 or alarm_saat_basi < 0:
        raise ValueError("toplam_saat > 0 ve alarm_saat_basi ≥ 0 olmalı")
    normal = s[y == 0]
    if normal.size == 0:
        raise ValueError("bütçe eşiği için normal örnek gerekli")
    fp_max = int(np.floor(alarm_saat_basi * toplam_saat + 1e-9))
    if fp_max <= 0:
        return float(np.nextafter(np.max(normal), np.inf))
    sirali = np.sort(normal)[::-1]
    kth = float(sirali[fp_max - 1])
    if int(np.sum(normal >= kth)) <= fp_max:
        return kth
    return float(np.nextafter(kth, np.inf))


def isletme_dosya(
    paket_val: dict[str, np.ndarray],
    paket_test: dict[str, np.ndarray],
    alarm_saat_basi: float,
) -> dict:
    """Tip başına bütçe eşiği (validasyon) → test olay oranı ve recall."""
    tp = fp = fn = 0
    normal_sn = 0.0
    for tip in np.unique(paket_test["tip"]):
        mte = paket_test["tip"] == tip
        mva = paket_val["tip"] == tip
        if not mva.any() or not np.any((paket_val["y"] == 0) & mva):
            continue
        saat_va = float(paket_val["sure_sn"][mva & (paket_val["y"] == 0)].sum()) / 3600.0
        if saat_va <= 0:
            continue
        esik = esik_fp_butcesi(
            paket_val["y"][mva], paket_val["skor"][mva], saat_va, alarm_saat_basi
        )
        met = MT.esikte_metrikler(paket_test["y"][mte], paket_test["skor"][mte], esik)
        tp += met["tp"]
        fp += met["fp"]
        fn += met["fn"]
        normal_sn += float(paket_test["sure_sn"][mte & (paket_test["y"] == 0)].sum())
    if tp + fp + fn == 0:
        raise ValueError("dosya işletme noktası seçilemedi")
    saat = normal_sn / 3600.0
    oran = (fp / saat) if saat > 0 else float("nan")
    return {
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fp": fp,
        "tp": tp,
        "fn": fn,
        "normal_saat": saat,
        "yanlis_alarm_saat": oran,
        "pr01": bool(saat > 0 and oran <= alarm_saat_basi + 1e-9),
    }


def dosya_auc(paket: dict[str, np.ndarray]) -> dict:
    """Dosya skorunda tip-içi AUC ortalaması ve havuz AUC."""
    auc_list = []
    tipler = {}
    y, s, tip = paket["y"], paket["skor"], paket["tip"]
    for t in np.unique(tip):
        m = tip == t
        if np.unique(y[m]).size < 2:
            continue
        a = MT.auc(y[m], s[m])
        tipler[str(t)] = a
        auc_list.append(a)
    if not auc_list:
        raise ValueError("dosya AUC için iki sınıflı tip yok")
    return {"makro_auc": float(np.mean(auc_list)), "havuz_auc": MT.auc(y, s), "tipler": tipler}
