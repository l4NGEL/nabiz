"""Resmi z ızgarası (2.0 ve üstü) ile kısa liste.

1.5 sigma önceki taramada validasyonda en yüksek recall'u verdi ama
görülmemiş koşulda bütçeyi taşıyamadı. Seçim, önceden tanımlı ızgaradadır.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.data.uyarlama import BirimOlcek  # noqa: E402
from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.eval.alarm import _birim_z, butce_sureklilik, dosya_auc  # noqa: E402
from nabiz.models.izleme import TipKnn, TipKuyruk, TipSpe  # noqa: E402
from nabiz.models.tip_dedektor import TipDedektoru  # noqa: E402

_spec = importlib.util.spec_from_file_location("en_iyi", Path(__file__).parent / "10_en_iyi.py")
_en = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_en)


def _birlestir(a: dict, b: dict, nasil: str) -> dict:
    za, zb = _birim_z(a)["skor"], _birim_z(b)["skor"]
    p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    p["skor"] = np.maximum(za, zb) if nasil == "max" else 0.5 * (za + zb)
    return p


def _karsilastir(veri) -> list[dict]:
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
    kurucular = [
        ("mahal", lambda: TipDedektoru()),
        ("spe64", lambda: TipSpe(64)),
        ("spe96", lambda: TipSpe(96)),
        ("spe120", lambda: TipSpe(120)),
        ("kuyruk32", lambda: TipKuyruk(32)),
        ("kuyruk64", lambda: TipKuyruk(64)),
        ("kuyruk96", lambda: TipKuyruk(96)),
        ("knn32", lambda: TipKnn(32)),
    ]
    ham = {}
    for ad, kur in kurucular:
        print(f"  egit {ad}", flush=True)
        model = kur().egit(Xtr, tip_tr[normal])
        ham[ad] = (
            model.anomali_skoru(Xva, tip_va),
            model.anomali_skoru(Xte, tip_te),
        )
    paket = {}
    satirlar = []

    def olc(ad: str, pval: dict, pte: dict) -> None:
        r = butce_sureklilik(pval, pte, MT.PR01_ALARM_SAAT)
        auc = dosya_auc(pte)
        satir = {
            "strateji": veri.strateji, "ad": ad, "val_recall": r["val_recall"],
            "uygun": r["uygun"], "z": r["esik"], "k": r["k"], "recall": r["recall"],
            "olay_fp": r["olay_fp"], "saat": r["normal_saat"], "pr01": r["pr01"],
            "auc": auc["makro_auc"],
            "oturum": _en._oturum_recall(pte, r["esik"], r["k"]),
        }
        satirlar.append(satir)
        _en._yaz(satir)

    for ad, (sva, ste) in ham.items():
        pval = _en._paket(sva, veri.y_val, veri.meta_val, "medyan")
        pte = _en._paket(ste, veri.y_test, veri.meta_test, "medyan")
        paket[ad] = (pval, pte)
        olc(ad, pval, pte)
    for ad, sol, sag, nasil in (
        ("mahal+spe64", "mahal", "spe64", "ort"),
        ("mahal+knn", "mahal", "knn32", "ort"),
        ("spe64+knn", "spe64", "knn32", "ort"),
        ("max(mahal,spe64)", "mahal", "spe64", "max"),
        ("max(spe64,knn)", "spe64", "knn32", "max"),
    ):
        pv = _birlestir(paket[sol][0], paket[sag][0], nasil)
        pt = _birlestir(paket[sol][1], paket[sag][1], nasil)
        olc(ad, pv, pt)
    return satirlar


def main() -> None:
    print(f"okunuyor {config.ISLENMIS_VERI / 'oznitelikler.parquet'}", flush=True)
    df = pd.read_parquet(config.ISLENMIS_VERI / "oznitelikler.parquet")
    hepsi = []
    for strateji in ("birim_disi", "rejim_holdout"):
        hepsi.extend(_karsilastir(_en._bol(df, strateji)))
    satirlar = []
    for s in hepsi:
        satirlar.append(
            f"| {s['strateji']} | {s['ad']} | {s['val_recall']:.3f} | {s['uygun']} | "
            f"{s['z']:.1f} | {s['k']} | {s['recall']:.3f} | {s['olay_fp']} | "
            f"{s['saat']:.2f} | {s['pr01']} | {s['auc']:.3f} | {s['oturum']:.3f} |"
        )
    govde = "\n".join([
        "# Resmi ızgara — z ≥ 2, eşit recall'da en uzun k",
        "",
        "Seçim yalnız validasyon. Test sütunları rapor içindir.",
        "",
        "| strateji | skor | val recall | uygun | z | k | test recall | FP olay | normal saat | PR-01 | dosya AUC | oturum recall |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|",
        *satirlar,
        "",
    ])
    hedef = Path("sonuclar") / "faz2-resmi-izgara.md"
    hedef.write_text(govde, encoding="utf-8")
    print(f"yazildi {hedef}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
