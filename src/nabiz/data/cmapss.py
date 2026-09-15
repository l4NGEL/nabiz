"""C-MAPSS (NASA türbofan bozulma simülasyonu) veri yükleme + sentetik üretim + pencereleme.

Gerçek format: `train_FD00X.txt` / `test_FD00X.txt` — boşlukla ayrılmış, başlıksız,
26 sütun: birim no, çevrim, 3 çalışma ayarı, 21 sensör. `RUL_FD00X.txt`: test
birimi başına, kesildiği noktadaki gerçek kalan ömür (tek sütun).

RUL etiketi: train'de run-to-failure (max_cevrim - cevrim); test'te
`RUL_FD00X.txt` + kalan (max_cevrim - cevrim). İkisinde de tavan uygulanır
(piecewise-linear RUL — literatürde standart: erken ömürde bozulma henüz
gözlenemez, tahmin edilebilir kabul edilmez).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SUTUNLAR = ["birim", "cevrim", "ayar_1", "ayar_2", "ayar_3"] + [f"sensor_{i}" for i in range(1, 22)]
SENSOR_SUTUNLARI = [f"sensor_{i}" for i in range(1, 22)]
TAVAN_RUL = 125.0  # piecewise-linear RUL tavanı (literatür standardı)


def train_oku(yol: str | Path) -> pd.DataFrame:
    """Gerçek C-MAPSS train_FD00X.txt oku."""
    df = pd.read_csv(yol, sep=r"\s+", header=None, names=SUTUNLAR)
    df.attrs["kaynak"] = "cmapss-gercek-train"
    return df


def test_oku(yol: str | Path, rul_yol: str | Path) -> pd.DataFrame:
    """Gerçek C-MAPSS test_FD00X.txt + RUL_FD00X.txt oku; `rul_test_sonu` ekler."""
    df = pd.read_csv(yol, sep=r"\s+", header=None, names=SUTUNLAR)
    rul_son = pd.read_csv(rul_yol, sep=r"\s+", header=None, names=["rul_test_sonu"])
    rul_son.index = rul_son.index + 1  # birim no 1-tabanlı
    df = df.merge(rul_son, left_on="birim", right_index=True, how="left")
    df.attrs["kaynak"] = "cmapss-gercek-test"
    return df


def sentetik_uret(
    n_birim: int = 60,
    min_cevrim: int = 120,
    maks_cevrim: int = 280,
    n_sensor: int = 21,
    n_bilgilendirici: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """C-MAPSS şemasında sahte run-to-failure trajectory'leri (gerçek dosya YOK).

    Her birim rastgele bir ömür L çeker; `n_bilgilendirici` sensör (D001'de
    olduğu gibi hepsi değil) çevrim ilerledikçe monoton kayar + gürültü;
    kalanlar düz gürültü. Aynı sütun şeması → downstream kod değişmeden çalışır.
    """
    rng = np.random.default_rng(seed)
    bilgilendirici = rng.choice(n_sensor, size=min(n_bilgilendirici, n_sensor), replace=False)
    yonler = {int(i): int(rng.choice([-1, 1])) for i in bilgilendirici}
    taban_genel = rng.normal(0, 1, n_sensor)

    satirlar: list[dict] = []
    for birim in range(1, n_birim + 1):
        L = int(rng.integers(min_cevrim, maks_cevrim + 1))
        taban = taban_genel + rng.normal(0, 0.3, n_sensor)  # birime özgü ofset
        ayar = rng.normal(0, 0.01, 3)
        for t in range(1, L + 1):
            ilerleme = t / L
            degerler = taban.copy()
            for i in bilgilendirici:
                degerler[i] += yonler[i] * (ilerleme**2) * rng.uniform(2.0, 5.0)
            degerler += rng.normal(0, 0.15, n_sensor)
            satir = {"birim": birim, "cevrim": t}
            satir.update({f"ayar_{k + 1}": float(ayar[k] + rng.normal(0, 0.005)) for k in range(3)})
            satir.update({f"sensor_{i + 1}": float(degerler[i]) for i in range(n_sensor)})
            satirlar.append(satir)

    df = pd.DataFrame(satirlar, columns=SUTUNLAR)
    df.attrs["kaynak"] = "cmapss-sentetik"
    df.attrs["bilgilendirici_sensorler"] = sorted(int(i) + 1 for i in bilgilendirici)
    return df


def birim_split(
    birimler: np.ndarray, test_orani: float = 0.2, val_orani: float = 0.1, seed: int = 0
) -> dict[int, str]:
    """Birimleri (motor ID) train/val/test'e ayır — hash tabanlı, deterministik,
    sıradan bağımsız. `data/splits.py`'deki desenle tutarlı."""
    benzersiz = sorted(set(int(b) for b in birimler))
    tuz = f"cmapss_birim:{seed}".encode()
    atama: dict[int, str] = {}
    for b in benzersiz:
        h = hashlib.sha1(tuz + str(b).encode()).digest()[:8]
        r = (int.from_bytes(h, "big") % 1_000_000) / 1_000_000.0
        if r < test_orani:
            atama[b] = "test"
        elif r < test_orani + val_orani:
            atama[b] = "val"
        else:
            atama[b] = "train"
    return atama


def pencerele(
    df: pd.DataFrame,
    pencere: int = 30,
    adim: int = 1,
    sensor_sutunlari: list[str] | None = None,
    tavan_rul: float = TAVAN_RUL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Birim başına kayan pencere: (pencere, n_sensor) -> X; pencere sonundaki
    kırpılmış RUL -> y. Döner: (X, y, birim_id) — hepsi aynı sırada."""
    sensor_sutunlari = sensor_sutunlari or SENSOR_SUTUNLARI
    X: list[np.ndarray] = []
    y: list[float] = []
    birim_id: list[int] = []

    for birim, grup in df.groupby("birim"):
        grup = grup.sort_values("cevrim")
        veri = grup[sensor_sutunlari].to_numpy(dtype=np.float32)
        if len(veri) < pencere:
            continue
        maks_cevrim = int(grup["cevrim"].max())
        rul_son = float(grup["rul_test_sonu"].iloc[0]) if "rul_test_sonu" in grup.columns else 0.0

        for son in range(pencere, len(veri) + 1, adim):
            cevrim_o_an = int(grup["cevrim"].iloc[son - 1])
            rul = rul_son + (maks_cevrim - cevrim_o_an)
            X.append(veri[son - pencere : son])
            y.append(min(rul, tavan_rul))
            birim_id.append(int(birim))

    if not X:
        raise ValueError(f"hiçbir birim {pencere} çevrimden uzun değil — pencere küçült")
    return np.stack(X), np.asarray(y, dtype=np.float32), np.asarray(birim_id, dtype=np.int64)


def son_pencere(
    df: pd.DataFrame,
    pencere: int = 30,
    sensor_sutunlari: list[str] | None = None,
    tavan_rul: float = TAVAN_RUL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Birim başına yalnız SON pencere + tek RUL — resmi C-MAPSS test protokolü
    (trajectory kesilme noktasında tek tahmin). Eğitim için `pencerele()`
    kullan (kaydırmalı, daha çok örnek); değerlendirme için bunu kullan —
    aksi halde örtüşen pencereler C-MAPSS skorunu suni şişirir."""
    sensor_sutunlari = sensor_sutunlari or SENSOR_SUTUNLARI
    X: list[np.ndarray] = []
    y: list[float] = []
    birim_id: list[int] = []

    for birim, grup in df.groupby("birim"):
        grup = grup.sort_values("cevrim")
        veri = grup[sensor_sutunlari].to_numpy(dtype=np.float32)
        if len(veri) < pencere:
            continue
        rul_son = float(grup["rul_test_sonu"].iloc[0]) if "rul_test_sonu" in grup.columns else 0.0
        X.append(veri[-pencere:])
        y.append(min(rul_son, tavan_rul))
        birim_id.append(int(birim))

    if not X:
        raise ValueError(f"hiçbir birim {pencere} çevrimden uzun değil — pencere küçült")
    return np.stack(X), np.asarray(y, dtype=np.float32), np.asarray(birim_id, dtype=np.int64)
