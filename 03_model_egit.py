"""Faz 2 — baseline'lar + dense autoencoder eğit, değerlendir, karşılaştır.

  a_ (akustik) + z_ (zarf) + t_ (telemetri) öznitelikleri üzerinde
  gözetimsiz anomali tespiti (yalnız normal ile eğitim).

Çıktı:  sonuclar/faz2-modeller.md   (strateji x model AUC/pAUC/F1/FPR + kırılım)

Kullanım:
  python 03_model_egit.py                       # birim_disi + rejim_holdout
  python 03_model_egit.py --strateji birim_ici
  python 03_model_egit.py --hizli                # AE epok azalt (duman testi)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import oznitelik as OZ  # noqa: E402
from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.models import baseline as B  # noqa: E402
from nabiz.models.temel import Olcekleyici  # noqa: E402

utf8_cikti()

try:
    from nabiz.models.autoencoder import DenseAE

    TORCH_VAR = True
except ImportError:
    DenseAE = None
    TORCH_VAR = False


def modelleri_kur(oznitelik_adlari: list[str], hizli: bool) -> list:
    modeller = [
        B.RMSEsigi(oznitelik_adlari),
        B.IzolasyonOrmani(seed=config.VARSAYILAN_SEED),
        B.Mahalanobis(),
    ]
    if TORCH_VAR:
        modeller.append(DenseAE(epok=8 if hizli else 120, seed=config.VARSAYILAN_SEED))
    return modeller


def strateji_calistir(strateji: str, hizli: bool) -> pd.DataFrame:
    veri = OZ.yukle(strateji)
    print(f"  {veri.ozet()}")

    olc = Olcekleyici().egit(veri.X_train_normal)
    Xtr_n = olc.uygula(veri.X_train_normal)
    Xva, Xte = olc.uygula(veri.X_val), olc.uygula(veri.X_test)
    saat = float((veri.y_test == 0).sum()) * config.PENCERE_SN / 3600.0

    satirlar = []
    for model in modelleri_kur(veri.oznitelik_adlari, hizli):
        t0 = time.perf_counter()
        try:
            # RMS ham ölçekte çalışır (a_rms özniteliği); diğerleri ölçekli
            if isinstance(model, B.RMSEsigi):
                model.egit(veri.X_train_normal)
                s_va, s_te = model.anomali_skoru(veri.X_val), model.anomali_skoru(veri.X_test)
            elif TORCH_VAR and isinstance(model, DenseAE):
                model.egit(Xtr_n, X_val=Xva[veri.y_val == 0] if (veri.y_val == 0).any() else None)
                s_va, s_te = model.anomali_skoru(Xva), model.anomali_skoru(Xte)
            else:
                model.egit(Xtr_n)
                s_va, s_te = model.anomali_skoru(Xva), model.anomali_skoru(Xte)

            rapor = MT.anomali_raporu(
                veri.y_test, s_te, veri.y_val, s_va,
                kirilim={"makine_tipi": veri.kirilim("makine_tipi")},
                toplam_saat=saat,
            )
        except ValueError as e:
            print(f"    {model.ad:18s} atlandı: {e}")
            continue
        satirlar.append(
            {
                "model": model.ad,
                "AUC": rapor["auc"],
                "AUC_CI": f"[{rapor['auc_alt']:.3f}, {rapor['auc_ust']:.3f}]",
                "pAUC": rapor["pauc"],
                "F1": rapor["esikte"]["f1"],
                "precision": rapor["esikte"]["precision"],
                "recall": rapor["esikte"]["recall"],
                "FPR": rapor["esikte"]["fpr"],
                "yanlış_alarm/sa": rapor.get("yanlis_alarm_saat", float("nan")),
                "sn": time.perf_counter() - t0,
                "_kirilim": rapor["kirilim"]["makine_tipi"],
            }
        )
        print(f"    {model.ad:18s} AUC {rapor['auc']:.3f}  pAUC {rapor['pauc']:.3f}  F1 {rapor['esikte']['f1']:.3f}")

    if not satirlar:
        raise ValueError("hiçbir model değerlendirilemedi (tek sınıflı test/val?)")
    df = pd.DataFrame(satirlar)
    df.insert(0, "strateji", strateji)
    return df


def rapor_yaz(tablolar: list[pd.DataFrame], hizli: bool) -> Path:
    config.SONUC_DIZINI.mkdir(parents=True, exist_ok=True)
    yol = config.SONUC_DIZINI / "faz2-modeller.md"
    goster = ["strateji", "model", "AUC", "AUC_CI", "pAUC", "F1", "precision", "recall", "FPR", "yanlış_alarm/sa", "sn"]
    parcalar = [
        "# Faz 2 — model karşılaştırması\n",
        "- öznitelik: a_ + z_ + t_  ·  gözetimsiz (yalnız normal eğitim)",
        f"- torch/AE: {'var' if TORCH_VAR else 'YOK — yalnız baseline'}"
        f"{'  (hızlı mod: AE epok=8)' if hizli and TORCH_VAR else ''}",
        "- NOT: sentetik veri — sayılar boru hattı doğrulaması içindir, "
        "portföy değeri için MIMII gerekir\n",
    ]
    tum = pd.concat(tablolar, ignore_index=True)
    parcalar.append(tum[goster].round(3).to_markdown(index=False))

    parcalar.append("\n## Makine tipine göre AUC kırılımı\n")
    for _, satir in tum.iterrows():
        k = satir["_kirilim"]
        if k:
            kirilim_str = ", ".join(f"{g}: {v['auc']:.3f} (n={v['n']})" for g, v in k.items())
            parcalar.append(f"- **{satir['strateji']} / {satir['model']}** — {kirilim_str}")

    if {"birim_disi", "rejim_holdout"} <= set(tum["strateji"]):
        parcalar.append("\n## Domain-shift (RB-01)\n")
        for model in tum["model"].unique():
            bd = tum[(tum.strateji == "birim_disi") & (tum.model == model)]["AUC"]
            rh = tum[(tum.strateji == "rejim_holdout") & (tum.model == model)]["AUC"]
            if len(bd) and len(rh):
                fark = float(bd.iloc[0] - rh.iloc[0])
                parcalar.append(f"- {model}: birim_disi {bd.iloc[0]:.3f} → rejim_holdout {rh.iloc[0]:.3f}  (Δ {fark:+.3f})")

    yol.write_text("\n".join(parcalar), encoding="utf-8")
    return yol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strateji", default=None, help="tek strateji (öntanım: birim_disi + rejim_holdout)")
    ap.add_argument("--hizli", action="store_true", help="AE epok azalt (duman testi)")
    a = ap.parse_args()

    config.dizinleri_hazirla()
    if not (config.ISLENMIS_VERI / "oznitelikler.parquet").exists():
        print("HATA: öznitelik yok. Önce:  python 02_oznitelik_cikar.py")
        return 1

    stratejiler = [a.strateji] if a.strateji else ["birim_disi", "rejim_holdout"]
    print(f"torch/AE: {'var' if TORCH_VAR else 'YOK'}")
    tablolar = []
    for s in stratejiler:
        print(f"\n[{s}]")
        try:
            tablolar.append(strateji_calistir(s, a.hizli))
        except ValueError as e:
            print(f"  atlandı: {e}")

    if not tablolar:
        print("HATA: hiçbir strateji çalışmadı")
        return 1
    yol = rapor_yaz(tablolar, a.hizli)
    print(f"\nyazildi -> {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
