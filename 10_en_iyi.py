"""Skor ailesi karşılaştırması. Eşik yalnız validasyondan seçilir.

Adaylar aynı birim ölçeğini paylaşır. Kazanan, bütçeyi validasyonda tutan
noktalar arasında en yüksek dosya recall'una sahip olandır. Test etiketi
seçime girmez.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.data import oznitelik as OZ  # noqa: E402
from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.eval.alarm import (  # noqa: E402
    _birim_z,
    alarm_olaylari,
    butce_sureklilik,
    dosya_auc,
    dosyalara_indir,
)
from nabiz.models.izleme import LedoitUzaklik, TipGmm, TipKnn, TipSpe, TipT2  # noqa: E402
from nabiz.models.tip_dedektor import TipDedektoru  # noqa: E402

Z_ADAY = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
GRUPLAR = ("a",)


def _bol(df: pd.DataFrame, strateji: str) -> OZ.BolmeVerisi:
    bol = f"bolme__{strateji}"
    oz = OZ.oznitelik_sutunlari(df, GRUPLAR)
    y = (df["etiket"] == "anomali").astype(int).to_numpy()
    maske = {k: (df[bol] == k).to_numpy() for k in ("train", "val", "test")}
    sutun = OZ._META + [c for c in OZ._META_OPSIYONEL if c in df.columns]

    def meta(k: str) -> pd.DataFrame:
        return df.loc[maske[k], sutun].reset_index(drop=True)

    return OZ.BolmeVerisi(
        strateji=strateji,
        oznitelik_adlari=oz,
        X_train=df.loc[maske["train"], oz].to_numpy(np.float64),
        y_train=y[maske["train"]],
        X_val=df.loc[maske["val"], oz].to_numpy(np.float64),
        y_val=y[maske["val"]],
        X_test=df.loc[maske["test"], oz].to_numpy(np.float64),
        y_test=y[maske["test"]],
        meta_test=meta("test"),
        meta_train=meta("train"),
        meta_val=meta("val"),
    )


def _paket(skor, y, meta, yontem: str) -> dict:
    return dosyalara_indir(
        skor, y, meta["dosya"].to_numpy(), meta["makine_tipi"].to_numpy(),
        meta["pencere_idx"].to_numpy(), yontem, config.PENCERE_SN, config.BINDIRME_SN,
        birim=meta["makine_birimi"].to_numpy(),
    )


def _oturum_recall(paket, esik: float, k: int) -> float:
    z = _birim_z(paket)
    hedef = set(z["oturum"][z["y"] == 1].tolist())
    if not hedef:
        return float("nan")
    yakalanan = {o["oturum"] for o in alarm_olaylari(z, esik, k)}
    return len(hedef & yakalanan) / len(hedef)


def _olc(ad: str, pval: dict, pte: dict) -> dict:
    r = butce_sureklilik(pval, pte, MT.PR01_ALARM_SAAT, z_aday=Z_ADAY)
    auc = dosya_auc(pte)
    return {
        "ad": ad,
        "val_recall": r["val_recall"],
        "uygun": r["uygun"],
        "z": r["esik"],
        "k": r["k"],
        "recall": r["recall"],
        "olay_fp": r["olay_fp"],
        "saat": r["normal_saat"],
        "pr01": r["pr01"],
        "auc": auc["makro_auc"],
        "oturum": _oturum_recall(pte, r["esik"], r["k"]),
    }


def _yaz(satir: dict) -> None:
    print(
        f"{satir['ad']:22} val={satir['val_recall']:.3f} uygun={satir['uygun']} "
        f"z={satir['z']:.1f} k={satir['k']} test_recall={satir['recall']:.3f} "
        f"fp={satir['olay_fp']} / {satir['saat']:.2f}h pr01={satir['pr01']} "
        f"auc={satir['auc']:.3f} oturum={satir['oturum']:.3f}",
        flush=True,
    )


def _adaylar(Xtr, tip, Xva, Xte, tip_va, tip_te):
    kurucular = [
        ("mahal", lambda: TipDedektoru()),
        ("ledoit", lambda: TipDedektoru(fabrika=LedoitUzaklik)),
        ("spe16", lambda: TipSpe(16)),
        ("spe32", lambda: TipSpe(32)),
        ("spe64", lambda: TipSpe(64)),
        ("t2_32", lambda: TipT2(32)),
        ("gmm16x2", lambda: TipGmm(16, 2)),
        ("knn32", lambda: TipKnn(32)),
    ]
    pencere = {}
    for ad, kur in kurucular:
        print(f"  egit {ad}", flush=True)
        model = kur().egit(Xtr, tip)
        pencere[ad] = (
            model.anomali_skoru(Xva, tip_va),
            model.anomali_skoru(Xte, tip_te),
        )
    return pencere


def _karsilastir(veri: OZ.BolmeVerisi) -> list[dict]:
    from nabiz.data.uyarlama import BirimOlcek

    print(f"\n== {veri.strateji} {veri.ozet()}", flush=True)
    birim_tr = veri.meta_train["makine_birimi"].to_numpy()
    tip_tr = veri.meta_train["makine_tipi"].to_numpy()
    normal = veri.y_train == 0
    olcek = BirimOlcek().egit(veri.X_train, birim_tr, veri.y_train)
    Xtr = olcek.uygula(veri.X_train[normal], birim_tr[normal], yeniden=False)
    Xva = olcek.uygula(veri.X_val, veri.meta_val["makine_birimi"].to_numpy(), yeniden=True)
    Xte = olcek.uygula(veri.X_test, veri.meta_test["makine_birimi"].to_numpy(), yeniden=True)
    tip_va = veri.meta_val["makine_tipi"].to_numpy()
    tip_te = veri.meta_test["makine_tipi"].to_numpy()
    pencere = _adaylar(Xtr, tip_tr[normal], Xva, Xte, tip_va, tip_te)

    satirlar = []
    paketler = {}
    for ad, (sva, ste) in pencere.items():
        for yontem in ("medyan", "tepe3"):
            if ad not in ("mahal", "spe32", "ledoit", "knn32") and yontem == "tepe3":
                continue
            pval = _paket(sva, veri.y_val, veri.meta_val, yontem)
            pte = _paket(ste, veri.y_test, veri.meta_test, yontem)
            paketler[f"{ad}/{yontem}"] = (pval, pte)
            satir = _olc(f"{ad}/{yontem}", pval, pte)
            satirlar.append(satir)
            _yaz(satir)

    for ad, sol, sag in (
        ("mahal+spe32", "mahal/medyan", "spe32/medyan"),
        ("mahal+knn", "mahal/medyan", "knn32/medyan"),
        ("spe32+t2", "spe32/medyan", "t2_32/medyan"),
    ):
        if sol not in paketler or sag not in paketler:
            continue
        birlesik = []
        for i in (0, 1):
            a, b = paketler[sol][i], paketler[sag][i]
            za, zb = _birim_z(a)["skor"], _birim_z(b)["skor"]
            p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
            p["skor"] = 0.5 * (za + zb)
            birlesik.append(p)
        satir = _olc(ad, birlesik[0], birlesik[1])
        satirlar.append(satir)
        _yaz(satir)
    return satirlar


def main() -> None:
    yol = config.ISLENMIS_VERI / "oznitelikler.parquet"
    print(f"okunuyor {yol}", flush=True)
    df = pd.read_parquet(yol)
    hepsi = []
    for strateji in ("birim_disi", "rejim_holdout"):
        hepsi.extend((strateji, s) for s in _karsilastir(_bol(df, strateji)))
    satirlar = []
    for strateji, s in hepsi:
        satirlar.append(
            f"| {strateji} | {s['ad']} | {s['val_recall']:.3f} | {s['uygun']} | "
            f"{s['z']:.1f} | {s['k']} | {s['recall']:.3f} | {s['olay_fp']} | "
            f"{s['saat']:.2f} | {s['pr01']} | {s['auc']:.3f} | {s['oturum']:.3f} |"
        )
    govde = "\n".join([
        "# Skor ailesi — validasyon seçimi",
        "",
        "Eşik ve k yalnız validasyondan. Test sayıları rapor içindir.",
        "",
        "| strateji | skor | val recall | uygun | z | k | test recall | FP olay | normal saat | PR-01 | dosya AUC | oturum recall |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|",
        *satirlar,
        "",
    ])
    hedef = Path("sonuclar") / "faz2-ayiricilar.md"
    hedef.parent.mkdir(exist_ok=True)
    hedef.write_text(govde, encoding="utf-8")
    print(f"yazildi {hedef}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
