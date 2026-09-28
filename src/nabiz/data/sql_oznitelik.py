"""C-MAPSS ölçümleri için SQL veri hattı: şema, yükleme, pencere-fonksiyonu öznitelikleri.

Sensör kayıtları `cmapss_olcum` tablosunda (kaynak, birim, cevrim) anahtarıyla
tutulur; eğitim öznitelikleri doğrudan SQL pencere fonksiyonlarıyla üretilir.
Aynı sorgu hem PostgreSQL'de (üretim) hem SQLite'ta (birim testi, CI) çalışır —
bu yüzden `regr_slope`/`STDDEV`/`SQRT`/`GREATEST` gibi lehçeye özgü fonksiyonlar
YOK; eğim ve varyans `SUM() OVER` toplamlarından kapalı formülle hesaplanır.

Tüm pencereler NEDENSELDİR (`ROWS ... PRECEDING AND CURRENT ROW`): bir
çevrimin öznitelikleri yalnız o çevrim ve öncesinden gelir — test motorunun son
çevriminde tahmin yapılırken geleceğe sızıntı olmaz.

Öznitelikler (seçilen her sensör için):
  ort{w}    son w çevrim ortalaması (w = 10, 30)
  egim{w}   son w çevrim üzerinde en küçük kareler eğimi (değer / çevrim)
  var30     son 30 çevrim örnek varyansı (ddof=1)
  sapma     ort10 - motorun kendi ilk 20 çevrim ortalaması (sağlıklı taban)
Pencere dolmamışsa (ilk çevrimler) mevcut satırlarla hesaplanır; tek satırda
tanımsız olan eğim/varyans NULL döner (NaN — HistGradientBoosting doğal işler).

`oznitelik_pandas()` aynı tanımın bağımsız pandas karşılığıdır; testler iki
yolun sayısal olarak eşleştiğini doğrular (SQL mantığı için referans).
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nabiz.data.cmapss import SUTUNLAR

TABLO = "cmapss_olcum"
RUL_TABLO = "cmapss_rul_son"
PENCERELER = (10, 30)
VAR_PENCERE = 30
TABAN_CEVRIM = 20
_OLCUM_SUTUNLARI = SUTUNLAR[2:]  # ayar_1..3 + sensor_1..21


@dataclass
class Baglanti:
    """DB-API bağlantısı + lehçe bilgisi (parametre işareti ve tip adları farklı)."""

    con: Any
    lehce: str  # "sqlite" | "postgresql"

    @property
    def isaret(self) -> str:
        return "?" if self.lehce == "sqlite" else "%s"

    def kapat(self) -> None:
        self.con.close()


def baglan(dsn: str) -> Baglanti:
    """`sqlite:///yol.db`, `sqlite://:memory:` veya `postgresql://...` DSN'inden bağlan."""
    if dsn.startswith("sqlite://"):
        import sqlite3

        yol = dsn[len("sqlite:///"):] if dsn.startswith("sqlite:///") else dsn[len("sqlite://"):]
        return Baglanti(sqlite3.connect(yol or ":memory:", check_same_thread=False), "sqlite")
    if dsn.startswith(("postgresql://", "postgres://")):
        import psycopg  # opsiyonel bağımlılık: pip install -e ".[sql]"

        return Baglanti(psycopg.connect(dsn), "postgresql")
    raise ValueError(f"desteklenmeyen DSN: {dsn!r} (sqlite:/// veya postgresql:// bekleniyor)")


def sema_olustur(b: Baglanti) -> None:
    reel = "REAL" if b.lehce == "sqlite" else "DOUBLE PRECISION"
    olcum = ", ".join(f"{s} {reel}" for s in _OLCUM_SUTUNLARI)
    cur = b.con.cursor()
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLO} (kaynak TEXT NOT NULL, birim INTEGER NOT NULL, "
        f"cevrim INTEGER NOT NULL, {olcum}, PRIMARY KEY (kaynak, birim, cevrim))"
    )
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {RUL_TABLO} (kaynak TEXT NOT NULL, birim INTEGER NOT NULL, "
        f"rul_test_sonu {reel} NOT NULL, PRIMARY KEY (kaynak, birim))"
    )
    b.con.commit()


def yukle(b: Baglanti, df: pd.DataFrame, kaynak: str) -> int:
    """Bir C-MAPSS DataFrame'ini `kaynak` etiketiyle yükle (idempotent: önce siler).

    `rul_test_sonu` sütunu varsa (test dosyası) birim başına `RUL_TABLO`'ya da yazılır.
    Döner: yüklenen ölçüm satırı sayısı.
    """
    eksik = [s for s in SUTUNLAR if s not in df.columns]
    if eksik:
        raise ValueError(f"eksik sütunlar: {eksik}")
    sutunlar = ["kaynak", *SUTUNLAR]
    yer = ", ".join([b.isaret] * len(sutunlar))
    satirlar = [
        (kaynak, int(r[0]), int(r[1]), *(float(v) for v in r[2:]))
        for r in df[SUTUNLAR].itertuples(index=False, name=None)
    ]
    cur = b.con.cursor()
    cur.execute(f"DELETE FROM {TABLO} WHERE kaynak = {b.isaret}", (kaynak,))
    cur.execute(f"DELETE FROM {RUL_TABLO} WHERE kaynak = {b.isaret}", (kaynak,))
    cur.executemany(f"INSERT INTO {TABLO} ({', '.join(sutunlar)}) VALUES ({yer})", satirlar)
    if "rul_test_sonu" in df.columns:
        rul = df.groupby("birim")["rul_test_sonu"].first()
        cur.executemany(
            f"INSERT INTO {RUL_TABLO} (kaynak, birim, rul_test_sonu) VALUES ({b.isaret}, {b.isaret}, {b.isaret})",
            [(kaynak, int(k), float(v)) for k, v in rul.items()],
        )
    b.con.commit()
    return len(satirlar)


def bilgilendirici_sensorler(df: pd.DataFrame, esik: float = 0.01) -> list[str]:
    """Standart sapması `esik`'i aşan sensörler. YALNIZ eğitim birimleriyle çağır.

    FD001'de sabit/neredeyse sabit 7 sensörü (1, 5, 6, 10, 16, 18, 19) eler —
    literatürdeki 14 sensörlük standart seçimle aynı sonucu veriyle türetir.
    """
    sd = df[[c for c in df.columns if c.startswith("sensor_")]].std()
    return [c for c in sd.index if sd[c] > esik]


def oznitelik_adlari(sensorler: Sequence[str]) -> list[str]:
    adlar = ["cevrim"]
    for s in sensorler:
        adlar += [f"{s}_ort{w}" for w in PENCERELER]
        adlar += [f"{s}_egim{w}" for w in PENCERELER]
        adlar += [f"{s}_var{VAR_PENCERE}", f"{s}_sapma"]
    return adlar


def oznitelik_sorgusu(
    sensorler: Sequence[str],
    isaret: str = "?",
    kaynak_filtresi: bool = True,
    birim_filtresi: bool = False,
    yalniz_son_cevrim: bool = False,
) -> str:
    """Pencere-fonksiyonu öznitelik SQL'i. Parametreler sırayla: kaynak, birim."""
    if not sensorler:
        raise ValueError("en az bir sensör gerekli")
    for s in sensorler:  # sütun adları SQL'e gömülür — yalnız bilinen adlar
        if s not in SUTUNLAR or not s.startswith("sensor_"):
            raise ValueError(f"bilinmeyen sensör: {s!r}")

    part = "PARTITION BY kaynak, birim ORDER BY cevrim"
    pencereler = {f"w{w}": f"({part} ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW)" for w in PENCERELER}
    pencereler["wtum"] = f"({part} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"

    # iç katman: pencere toplamları (eğim/varyans kapalı formülleri için)
    ic: list[str] = ["kaynak", "birim", "cevrim"]
    for w in PENCERELER:
        ic += [
            f"COUNT(*) OVER w{w} AS n{w}",
            f"SUM(cevrim) OVER w{w} AS st{w}",
            f"SUM(cevrim * cevrim) OVER w{w} AS stt{w}",
        ]
    for s in sensorler:
        for w in PENCERELER:
            ic += [
                f"SUM({s}) OVER w{w} AS {s}_sx{w}",
                f"SUM(cevrim * {s}) OVER w{w} AS {s}_stx{w}",
            ]
        ic.append(f"SUM({s} * {s}) OVER w{VAR_PENCERE} AS {s}_sxx{VAR_PENCERE}")
        ic.append(f"AVG(CASE WHEN cevrim <= {TABAN_CEVRIM} THEN {s} END) OVER wtum AS {s}_taban")

    kosullar = []
    if kaynak_filtresi:
        kosullar.append(f"kaynak = {isaret}")
    if birim_filtresi:
        kosullar.append(f"birim = {isaret}")
    where = f" WHERE {' AND '.join(kosullar)}" if kosullar else ""
    window = ", ".join(f"{ad} AS {tanim}" for ad, tanim in pencereler.items())

    # dış katman: öznitelikler. NULLIF: tek satırlık pencerede 0'a bölme -> NULL
    # (PostgreSQL 0'a bölmede hata verir, SQLite NULL döner — NULLIF ikisini eşitler).
    dis: list[str] = ["kaynak", "birim", "cevrim"]
    for s in sensorler:
        for w in PENCERELER:
            dis.append(f"{s}_sx{w} * 1.0 / n{w} AS {s}_ort{w}")
        for w in PENCERELER:
            dis.append(
                f"(n{w} * {s}_stx{w} - st{w} * {s}_sx{w}) * 1.0 "
                f"/ NULLIF(n{w} * stt{w} - st{w} * st{w}, 0) AS {s}_egim{w}"
            )
        v = VAR_PENCERE
        ham_var = f"({s}_sxx{v} - {s}_sx{v} * {s}_sx{v} * 1.0 / n{v}) / NULLIF(n{v} - 1, 0)"
        # kayan noktada ~0 varyans hafif negatif çıkabilir -> 0'a kırp
        dis.append(f"CASE WHEN {ham_var} < 0 THEN 0.0 ELSE {ham_var} END AS {s}_var{v}")
        dis.append(f"{s}_sx10 * 1.0 / n10 - {s}_taban AS {s}_sapma")

    sorgu = (
        f"SELECT {', '.join(dis)} FROM ("
        f"SELECT {', '.join(ic)} FROM {TABLO}{where} WINDOW {window}"
        f") toplamlar"
    )
    if yalniz_son_cevrim:
        sorgu = (
            f"SELECT * FROM (SELECT f.*, ROW_NUMBER() OVER "
            f"(PARTITION BY kaynak, birim ORDER BY cevrim DESC) AS sira FROM ({sorgu}) f) s "
            f"WHERE sira = 1"
        )
    return sorgu + " ORDER BY kaynak, birim, cevrim"


def oznitelik_cek(
    b: Baglanti,
    sensorler: Sequence[str],
    kaynak: str | None,
    birim: int | None = None,
    yalniz_son_cevrim: bool = False,
) -> pd.DataFrame:
    sorgu = oznitelik_sorgusu(
        sensorler, b.isaret, kaynak_filtresi=kaynak is not None,
        birim_filtresi=birim is not None, yalniz_son_cevrim=yalniz_son_cevrim,
    )
    params: list[Any] = [p for p in (kaynak, birim) if p is not None]
    cur = b.con.cursor()
    cur.execute(sorgu, params)
    adlar = [d[0] for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=adlar)
    df = df.drop(columns=["sira"], errors="ignore")
    oz = [c for c in df.columns if c not in ("kaynak", "birim", "cevrim")]
    df[oz] = df[oz].astype(float)  # NULL -> NaN
    return df


def sorgu_suresi(b: Baglanti, sensorler: Sequence[str], kaynak: str | None, tekrar: int = 3) -> float:
    """Öznitelik sorgusunun (tüm satırlar çekilerek) en iyi duvar-saati süresi, saniye."""
    en_iyi = float("inf")
    for _ in range(tekrar):
        t0 = time.perf_counter()
        oznitelik_cek(b, sensorler, kaynak)
        en_iyi = min(en_iyi, time.perf_counter() - t0)
    return en_iyi


def satir_sayilari(b: Baglanti) -> dict[str, int]:
    cur = b.con.cursor()
    cur.execute(f"SELECT kaynak, COUNT(*) FROM {TABLO} GROUP BY kaynak ORDER BY kaynak")
    return {str(k): int(n) for k, n in cur.fetchall()}


def test_rul_son(b: Baglanti, kaynak: str) -> pd.Series:
    cur = b.con.cursor()
    cur.execute(f"SELECT birim, rul_test_sonu FROM {RUL_TABLO} WHERE kaynak = {b.isaret}", (kaynak,))
    return pd.Series({int(k): float(v) for k, v in cur.fetchall()}, name="rul_test_sonu").sort_index()


# ------------------------------------------------------------------ referans


def _kayan_toplam(seri: pd.Series, w: int) -> pd.Series:
    return seri.rolling(w, min_periods=1).sum()


def oznitelik_pandas(df: pd.DataFrame, sensorler: Iterable[str]) -> pd.DataFrame:
    """`oznitelik_sorgusu()` ile AYNI tanımın bağımsız pandas uygulaması (test referansı)."""
    sensorler = list(sensorler)
    parcalar = []
    for birim, g in df.sort_values(["birim", "cevrim"]).groupby("birim", sort=True):
        t = g["cevrim"].astype(float)
        cikti = {"birim": np.full(len(g), int(birim)), "cevrim": g["cevrim"].to_numpy()}
        for s in sensorler:
            x = g[s].astype(float)
            for w in PENCERELER:
                cikti[f"{s}_ort{w}"] = x.rolling(w, min_periods=1).mean().to_numpy()
            for w in PENCERELER:
                n = t.rolling(w, min_periods=1).count()
                st, stt = _kayan_toplam(t, w), _kayan_toplam(t * t, w)
                sx, stx = _kayan_toplam(x, w), _kayan_toplam(t * x, w)
                payda = (n * stt - st * st).replace(0, np.nan)
                cikti[f"{s}_egim{w}"] = ((n * stx - st * sx) / payda).to_numpy()
            var = x.rolling(VAR_PENCERE, min_periods=1).var(ddof=1).clip(lower=0)
            cikti[f"{s}_var{VAR_PENCERE}"] = var.to_numpy()
            taban = x.where(g["cevrim"] <= TABAN_CEVRIM).expanding().mean()
            cikti[f"{s}_sapma"] = (x.rolling(10, min_periods=1).mean() - taban).to_numpy()
        parcalar.append(pd.DataFrame(cikti))
    return pd.concat(parcalar, ignore_index=True)
