"""Ürün dedektörü — akustik PCA artığı, dosya alarmı.

Skor, tip başına SPE (70 bileşen korunur, kalan artık enerjidir). Eşik (z) ve
süreklilik (k) yalnız validasyon bölümünden seçilir. Test etiketine bakılmaz.
Her makinenin dosya skorları kendi medyan/MAD'ine bölünür; alarm, bir kayıt
klasöründe k ardışık dosya eşiği geçince bir kez açılır.
"""
from __future__ import annotations

import numpy as np

from nabiz import config
from nabiz.data.oznitelik import BolmeVerisi
from nabiz.data.uyarlama import BirimOlcek
from nabiz.eval import metrics as MT
from nabiz.eval.alarm import _birim_z, alarm_olaylari, dosyalara_indir, isletme_sec
from nabiz.models.izleme import TipSpe


def _alarm_dosyalari(paket: dict[str, np.ndarray], esik: float, k: int) -> set[str]:
    """Eşik üst üste k kez tutunca, o dizideki bütün dosyalar alarmdadır."""
    oturum = paket["oturum"]
    dosya = paket["dosya"]
    skor = paket["skor"]
    icinde: set[str] = set()
    for ot in np.unique(oturum):
        m = np.flatnonzero(oturum == ot)
        m = m[np.argsort(dosya[m], kind="mergesort")]
        dizi: list[str] = []
        for i in m:
            if skor[i] >= esik:
                dizi.append(str(dosya[i]))
                if len(dizi) >= k:
                    icinde.update(dizi)
            else:
                dizi = []
    return icinde


class AkustikUrun:
    """Eğitilmiş ürün. `olaylar` ground-truth kullanmaz."""

    def __init__(self, n_bilesen: int = 70) -> None:
        self.n_bilesen = n_bilesen
        self.olcek: BirimOlcek | None = None
        self.model: TipSpe | None = None
        self.isletme: dict | None = None

    def egit(self, veri: BolmeVerisi) -> AkustikUrun:
        birim = veri.meta_train["makine_birimi"].to_numpy()
        tip = veri.meta_train["makine_tipi"].to_numpy()
        normal = veri.y_train == 0
        self.olcek = BirimOlcek().egit(veri.X_train, birim, veri.y_train)
        Xtr = self.olcek.uygula(veri.X_train[normal], birim[normal], yeniden=False)
        self.model = TipSpe(self.n_bilesen).egit(Xtr, tip[normal])
        self.isletme = isletme_sec(self._paket(veri, "val"), MT.PR01_ALARM_SAAT)
        return self

    def olaylar(self, veri: BolmeVerisi) -> list[dict]:
        if self.model is None or self.olcek is None or self.isletme is None:
            raise RuntimeError("AkustikUrun.egit() çağrılmadı")
        paket = _birim_z(self._paket(veri, "test"))
        return alarm_olaylari(paket, self.isletme["esik"], self.isletme["k"])

    def kayitlar(self, veri: BolmeVerisi) -> list[dict]:
        """Testteki her dosya. Karar etikete bakmaz; klasör yolu yalnız gösterim içindir."""
        if self.model is None or self.olcek is None or self.isletme is None:
            raise RuntimeError("AkustikUrun.egit() çağrılmadı")
        paket = _birim_z(self._paket(veri, "test"))
        esik = float(self.isletme["esik"])
        k = int(self.isletme["k"])
        alarmda = _alarm_dosyalari(paket, esik, k)
        satirlar = []
        for i, dosya in enumerate(paket["dosya"]):
            oturum = str(paket["oturum"][i]).replace("\\", "/")
            parca = set(oturum.split("/"))
            if "abnormal" in parca:
                klasor = "ariza"
            elif "normal" in parca:
                klasor = "saglam"
            else:
                klasor = "diger"
            skor = float(paket["skor"][i])
            ad = str(dosya).replace("\\", "/").rsplit("/", 1)[-1]
            satirlar.append({
                "dosya": ad,
                "birim": str(paket["birim"][i]),
                "klasor": klasor,
                "skor": skor,
                "sicak": skor >= esik,
                "alarm": str(dosya) in alarmda,
            })
        return satirlar

    def _paket(self, veri: BolmeVerisi, bolme: str) -> dict[str, np.ndarray]:
        if bolme == "val":
            X, y, meta = veri.X_val, veri.y_val, veri.meta_val
        elif bolme == "test":
            X, y, meta = veri.X_test, veri.y_test, veri.meta_test
        else:
            raise ValueError(f"bolme {bolme!r}")
        birim = meta["makine_birimi"].to_numpy()
        tip = meta["makine_tipi"].to_numpy()
        Xs = self.olcek.uygula(X, birim, yeniden=True)
        skor = self.model.anomali_skoru(Xs, tip)
        return dosyalara_indir(
            skor, y, meta["dosya"].to_numpy(), tip, meta["pencere_idx"].to_numpy(),
            "medyan", config.PENCERE_SN, config.BINDIRME_SN, birim=birim,
        )
