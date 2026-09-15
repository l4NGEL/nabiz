"""Faz 2 — RUL (kalan faydalı ömür) regresyonu: C-MAPSS.

Gerçek C-MAPSS varsa (data/ham/cmapss/train_FD00X.txt) onu, yoksa
data/ham/cmapss/train_sentetik.csv (python 00_veri_indir.py --cmapss-sentetik),
o da yoksa bellekte sentetik üretir.

Değerlendirme İKİ şekilde: kaydırmalı pencere (çok örnek, eğitim-tarzı — bilgi
amaçlı) VE resmi protokol (birim başına tek son pencere — literatürle
karşılaştırılabilir sayı; C-MAPSS skoru bunun üzerinden hesaplanır).

Çıktı:  sonuclar/faz2-rul.md

Kullanım:
  python 05_rul_egit.py
  python 05_rul_egit.py --hizli     # az epok (duman testi)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import cmapss as C  # noqa: E402
from nabiz.eval import metrics as MT  # noqa: E402

utf8_cikti()

try:
    from nabiz.models.rul import RULRegresor

    TORCH_VAR = True
except ImportError:
    TORCH_VAR = False

SEED = config.VARSAYILAN_SEED
PENCERE = 30


def veri_yukle() -> tuple:
    kok = config.HAM_VERI / "cmapss"
    gercek = sorted(kok.glob("train_FD*.txt")) if kok.exists() else []
    if gercek:
        train_yol = gercek[0]
        ad = train_yol.stem.replace("train_", "")
        test_yol = kok / f"test_{ad}.txt"
        rul_yol = kok / f"RUL_{ad}.txt"
        if test_yol.exists() and rul_yol.exists():
            return C.train_oku(train_yol), C.test_oku(test_yol, rul_yol), f"gercek-{ad}"
        return C.train_oku(train_yol), None, f"gercek-{ad}-yalniz-train"

    sentetik_yol = kok / "train_sentetik.csv"
    if sentetik_yol.exists():
        df = pd.read_csv(sentetik_yol)
        df.attrs["kaynak"] = "sentetik-disk"
        return df, None, "sentetik-disk"

    return C.sentetik_uret(seed=SEED), None, "sentetik-bellek"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hizli", action="store_true", help="az epok (duman testi)")
    ap.add_argument("--pencere", type=int, default=PENCERE)
    a = ap.parse_args()

    if not TORCH_VAR:
        print("HATA: torch yok.  pip install -e \".[dl]\"")
        return 1

    config.dizinleri_hazirla()
    train_df, test_df, kaynak = veri_yukle()
    print(f"kaynak={kaynak}, {train_df['birim'].nunique()} birim (train), {len(train_df)} satır")

    if test_df is not None:
        # gerçek C-MAPSS: train birimleri eğitim/val, test dosyası resmi test
        atama = C.birim_split(train_df["birim"].to_numpy(), test_orani=0.0, val_orani=0.15, seed=SEED)
        train_birim = {b for b, s in atama.items() if s == "train"}
        val_birim = {b for b, s in atama.items() if s == "val"}
        tr_df = train_df[train_df["birim"].isin(train_birim)]
        va_df = train_df[train_df["birim"].isin(val_birim)]
        te_df = test_df
    else:
        atama = C.birim_split(train_df["birim"].to_numpy(), seed=SEED)
        train_df = train_df.copy()
        train_df["_bolme"] = train_df["birim"].map(atama)
        tr_df = train_df[train_df["_bolme"] == "train"]
        va_df = train_df[train_df["_bolme"] == "val"]
        te_df = train_df[train_df["_bolme"] == "test"]

    Xtr, ytr, _ = C.pencerele(tr_df, pencere=a.pencere)
    Xva, yva, _ = C.pencerele(va_df, pencere=a.pencere)
    Xte_kaydir, yte_kaydir, _ = C.pencerele(te_df, pencere=a.pencere)
    Xte_son, yte_son, birim_te = C.son_pencere(te_df, pencere=a.pencere)
    print(f"pencere={a.pencere}: train {Xtr.shape}, val {Xva.shape}, "
          f"test-kaydirmali {Xte_kaydir.shape}, test-resmi {Xte_son.shape} ({len(birim_te)} birim)")

    mu, sd = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1))
    sd[sd < 1e-6] = 1.0

    def olcekle(X):
        return (X - mu) / sd

    model = RULRegresor(epok=15 if a.hizli else 150, seed=SEED)
    model.egit(olcekle(Xtr), ytr, olcekle(Xva), yva)
    print(f"eğitim: {len(model.gecmis_)} epok, son (normalize) kayıp {model.gecmis_[-1]:.4f}")

    pred_son = model.tahmin(olcekle(Xte_son))
    pred_kaydir = model.tahmin(olcekle(Xte_kaydir))

    taban_tahmin = np.full_like(yte_son, ytr.mean())  # baseline: her zaman train ortalaması

    def metrikler(y, pred):
        return {
            "MAE": float(np.mean(np.abs(pred - y))),
            "RMSE": float(np.sqrt(np.mean((pred - y) ** 2))),
            "cmapss_skoru": MT.cmapss_skoru(y, pred),
        }

    m_model = metrikler(yte_son, pred_son)
    m_taban = metrikler(yte_son, taban_tahmin)
    m_kaydir = metrikler(yte_kaydir, pred_kaydir)

    print(f"[resmi protokol, n={len(yte_son)}] rul_gru   MAE {m_model['MAE']:.2f}  "
          f"RMSE {m_model['RMSE']:.2f}  C-MAPSS {m_model['cmapss_skoru']:.1f}")
    print(f"[resmi protokol, n={len(yte_son)}] ort_taban MAE {m_taban['MAE']:.2f}  "
          f"RMSE {m_taban['RMSE']:.2f}  C-MAPSS {m_taban['cmapss_skoru']:.1f}")

    config.SONUC_DIZINI.mkdir(parents=True, exist_ok=True)
    yol = config.SONUC_DIZINI / "faz2-rul.md"
    gecti = m_model["MAE"] < m_taban["MAE"]
    yol.write_text(
        "\n".join(
            [
                "# Faz 2 — RUL (kalan faydalı ömür) regresyonu\n",
                f"- kaynak: `{kaynak}`  ·  pencere: {a.pencere} çevrim  ·  "
                f"train birim: {tr_df['birim'].nunique()}, val: {va_df['birim'].nunique()}, "
                f"test: {te_df['birim'].nunique()}",
                "- değerlendirme: resmi protokol = birim başına yalnız son pencere "
                "(literatürle karşılaştırılabilir); kaydırmalı = eğitim-tarzı, bilgi amaçlı\n",
                "## Resmi protokol (birim başına 1 tahmin)\n",
                "| model | MAE (çevrim) | RMSE | C-MAPSS skoru (düşük iyi) |",
                "|---|---:|---:|---:|",
                f"| rul_gru | {m_model['MAE']:.2f} | {m_model['RMSE']:.2f} | {m_model['cmapss_skoru']:.1f} |",
                f"| ort_taban (train ortalaması) | {m_taban['MAE']:.2f} | {m_taban['RMSE']:.2f} | "
                f"{m_taban['cmapss_skoru']:.1f} |",
                f"\n**Baseline'ı geçti: {'EVET' if gecti else 'HAYIR'}** (MAE karşılaştırması)\n",
                "## Kaydırmalı pencere (bilgi amaçlı, çok örnek)\n",
                f"n={len(yte_kaydir)} · MAE {m_kaydir['MAE']:.2f} · RMSE {m_kaydir['RMSE']:.2f} · "
                f"C-MAPSS skoru {m_kaydir['cmapss_skoru']:.1f} "
                "(örtüşen pencereler nedeniyle skor resmi protokolden daha büyük görünür — beklenen)\n",
                "NOT: sentetik veride sonuçlar boru hattı doğrulaması içindir; "
                "gerçek C-MAPSS (python 00_veri_indir.py --cmapss-bilgi) ile literatürle "
                "karşılaştırılabilir.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nyazildi -> {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
