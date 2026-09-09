"""Çok-kanal hizalama — FR-01.

Akustik + titreşim kanallarını ortak örnekleme hızına ve ortak zaman eksenine
getirir. Telemetri (düşük hızlı) ayrı ele alınır: `telemetri_hizala` ile hedef
zaman damgalarına interpolasyon.

Kabul kriteri (docs/gereksinimler.md FR-01): kayma < 1 ms; eksik kanal graceful.
`hizala` tamsayı örnek ofseti kullanır → alt-örnek hatası 0.5/hedef_hiz sn;
16 kHz+ hedefte < 1 ms (assert edilir).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly

from nabiz.dsp.dogrulama import pozitif, sonlu_1b


class HizalamaHatasi(ValueError):
    """Kanallar hizalanamadı (ortak aralık yok, tolerans aşıldı, vb.)."""


@dataclass
class Kanal:
    veri: np.ndarray
    ornekleme_hizi: float
    baslangic_sn: float = 0.0
    ad: str = ""

    def __post_init__(self) -> None:
        self.veri = sonlu_1b(self.veri, self.ad or "kanal")
        self.ornekleme_hizi = pozitif(self.ornekleme_hizi, "ornekleme_hizi")

    @property
    def sure_sn(self) -> float:
        return self.veri.size / self.ornekleme_hizi

    @property
    def bitis_sn(self) -> float:
        return self.baslangic_sn + self.sure_sn


def yeniden_ornekle(kanal: Kanal, hedef_hiz: float) -> Kanal:
    """Kanalı hedef örnekleme hızına getir (polyphase). Hız aynıysa kopya döner."""
    hedef_hiz = pozitif(hedef_hiz, "hedef_hiz")
    if abs(kanal.ornekleme_hizi - hedef_hiz) < 1e-6:
        return Kanal(kanal.veri.copy(), hedef_hiz, kanal.baslangic_sn, kanal.ad)
    kaynak = int(round(kanal.ornekleme_hizi))
    hedef = int(round(hedef_hiz))
    b = gcd(kaynak, hedef)
    yeni = resample_poly(kanal.veri, hedef // b, kaynak // b)
    return Kanal(np.asarray(yeni, dtype=np.float64), hedef_hiz, kanal.baslangic_sn, kanal.ad)


@dataclass
class HizalamaSonucu:
    kanallar: dict[str, np.ndarray]
    zaman_sn: np.ndarray
    hedef_hiz: float
    eksik_kanallar: list[str] = field(default_factory=list)
    hizalama_hatasi_sn: float = 0.0
    ortak_baslangic_sn: float = 0.0

    @property
    def ornek_sayisi(self) -> int:
        return self.zaman_sn.size


def hizala(
    kanallar: dict[str, Kanal | None],
    hedef_hiz: float,
    tolerans_sn: float = 1e-3,
) -> HizalamaSonucu:
    """Kanalları ortak hıza + ortak zaman aralığına indir.

    `None` veya boş kanallar `eksik_kanallar`'a yazılır ve atlanır (graceful).
    """
    hedef_hiz = pozitif(hedef_hiz, "hedef_hiz")
    eksik = sorted(
        ad
        for ad, k in kanallar.items()
        if k is None or np.asarray(k.veri).size == 0
    )
    aktif = {ad: k for ad, k in kanallar.items() if ad not in eksik}
    if not aktif:
        raise HizalamaHatasi("hizalanacak aktif kanal yok")

    yeniden = {ad: yeniden_ornekle(k, hedef_hiz) for ad, k in aktif.items()}
    bas = max(k.baslangic_sn for k in yeniden.values())
    son = min(k.bitis_sn for k in yeniden.values())
    if son - bas <= tolerans_sn:
        raise HizalamaHatasi(
            f"kanalların ortak zaman aralığı yok (bas={bas:.4f}s, son={son:.4f}s)"
        )

    n = int(np.floor((son - bas) * hedef_hiz))
    cikti: dict[str, np.ndarray] = {}
    hatalar: list[float] = []
    for ad, k in yeniden.items():
        tam_ofset = (bas - k.baslangic_sn) * hedef_hiz
        ofset = int(round(tam_ofset))
        hatalar.append(abs(tam_ofset - ofset) / hedef_hiz)
        cikti[ad] = np.asarray(k.veri[ofset : ofset + n], dtype=np.float64)

    hata = max(hatalar)
    if hata > tolerans_sn:
        raise HizalamaHatasi(
            f"hizalama hatası {hata * 1e3:.3f} ms > tolerans {tolerans_sn * 1e3:.3f} ms "
            f"(hedef_hiz {hedef_hiz:.0f} Hz çok düşük)"
        )

    return HizalamaSonucu(
        kanallar=cikti,
        zaman_sn=bas + np.arange(n) / hedef_hiz,
        hedef_hiz=hedef_hiz,
        eksik_kanallar=eksik,
        hizalama_hatasi_sn=hata,
        ortak_baslangic_sn=float(bas),
    )


def capraz_korelasyon_kaymasi(
    a, b, ornekleme_hizi: float, maks_kayma_sn: float = 0.1
) -> float:
    """`b`'nin `a`'ya göre gecikmesini (sn) çapraz korelasyonla kestir.

    Pozitif dönüş: b, a'dan bu kadar geç başlıyor. Ham kayıt eş-zamanlama
    kontrolü için (paylaşılan bir geçici / darbe varsa).
    """
    a = sonlu_1b(a, "a") - np.mean(a)
    b = sonlu_1b(b, "b") - np.mean(b)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    maks_lag = max(1, int(maks_kayma_sn * ornekleme_hizi))
    korr = correlate(a, b, mode="full")
    lag = correlation_lags(a.size, b.size, mode="full")
    pencere = np.abs(lag) <= maks_lag
    en_iyi = int(lag[pencere][np.argmax(korr[pencere])])
    # correlate zirvesi a[n] ile b[n-lag] hizalanınca; b, a'dan geç ise lag < 0.
    # Dışa "b'nin gecikmesi" (pozitif = geç) olarak veriyoruz → işareti çevir.
    return -en_iyi / ornekleme_hizi


def telemetri_hizala(kaynak_zaman, kaynak_deger, hedef_zaman) -> np.ndarray:
    """Düşük hızlı telemetriyi hedef zaman damgalarına interpola (uçlarda sabit)."""
    kz = np.asarray(kaynak_zaman, dtype=np.float64)
    kd = np.asarray(kaynak_deger, dtype=np.float64)
    hz = np.asarray(hedef_zaman, dtype=np.float64)
    if kz.size != kd.size:
        raise ValueError(f"kaynak_zaman ({kz.size}) ve kaynak_deger ({kd.size}) uzunluğu farklı")
    if kz.size < 2:
        raise ValueError("telemetri hizalaması için en az 2 örnek gerekli")
    if not np.all(np.isfinite(kz)) or not np.all(np.isfinite(kd)):
        raise ValueError("telemetri kaynağı sonlu değil")
    if np.any(np.diff(kz) <= 0):
        sira = np.argsort(kz, kind="stable")
        kz, kd = kz[sira], kd[sira]
    return np.interp(hz, kz, kd)
