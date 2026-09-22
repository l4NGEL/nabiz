"""Faz 2 — gözetimsiz anomali modelleri: eğit, değerlendir, karşılaştır.

Öznitelik-vektörü yolu (a_ + z_ + t_):  RMS eşiği, IsolationForest, Mahalanobis, DenseAE
Mel-tensör yolu (data/islenmis/mel.npz varsa):  ConvAE, CRNNAE, SpekTransformerAE
Füzyon (hibrid, aynı koşulda):  FusionAE — spektrogram-CNN ⊕ telemetri-MLP

Hepsi yalnız normal örneklerle eğitilir; anomali skoru = yeniden yapılandırma /
mesafe. Eşik validasyondan (F1) seçilir.

Çıktı:  sonuclar/faz2-modeller.md

Kullanım:
  python 03_model_egit.py                 # birim_disi + rejim_holdout, mevcut tüm modeller
  python 03_model_egit.py --strateji birim_ici
  python 03_model_egit.py --hizli          # AE epok az (duman testi)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import oznitelik as OZ  # noqa: E402
from nabiz.eval import kalibrasyon as KAL  # noqa: E402
from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.eval import ood as OOD  # noqa: E402
from nabiz.models import baseline as B  # noqa: E402
from nabiz.models.temel import Olcekleyici  # noqa: E402

utf8_cikti()

try:
    from nabiz.data import mel as MEL
    from nabiz.models.autoencoder import DenseAE
    from nabiz.models.cnn import ConvAE
    from nabiz.models.crnn import CRNNAE
    from nabiz.models.fusion import FusionAE
    from nabiz.models.transformer import SpekTransformerAE

    TORCH_VAR = True
except ImportError:
    TORCH_VAR = False

SEED = config.VARSAYILAN_SEED


def _rapor_satiri(ad: str, rapor: dict, sn: float) -> dict:
    return {
        "model": ad,
        "AUC": rapor["auc"],
        "AUC_CI": f"[{rapor['auc_alt']:.3f}, {rapor['auc_ust']:.3f}]",
        "pAUC": rapor["pauc"],
        "F1": rapor["esikte"]["f1"],
        "precision": rapor["esikte"]["precision"],
        "recall": rapor["esikte"]["recall"],
        "FPR": rapor["esikte"]["fpr"],
        "yanlış_alarm/sa": rapor.get("yanlis_alarm_saat", float("nan")),
        "sn": sn,
        "_kirilim": rapor["kirilim"]["makine_tipi"],
    }


def _degerlendir(ad, y_te, s_te, y_va, s_va, kirilim, saat, sn) -> dict | None:
    try:
        rapor = MT.anomali_raporu(
            y_te, s_te, y_va, s_va, kirilim={"makine_tipi": kirilim}, toplam_saat=saat
        )
    except ValueError as e:
        print(f"    {ad:20s} atlandı: {e}")
        return None
    print(f"    {ad:20s} AUC {rapor['auc']:.3f}  pAUC {rapor['pauc']:.3f}  F1 {rapor['esikte']['f1']:.3f}")
    return _rapor_satiri(ad, rapor, sn)


def _kalibrasyon_ood_calistir(ad: str, model, Xte, y_test, Xva, y_val, s_te, s_va) -> dict | None:
    """Val ile Platt kalibrasyonu (ECE öncesi/sonrası) + OOD ayrım gücü (gürültü +
    öznitelikler-arası korelasyonu bozulmuş girdi). RB — 'kendinden emin yanlış' kontrolü."""
    try:
        kal = KAL.PlattKalibratoru().egit(s_va, y_val)
    except ValueError as e:
        print(f"    kalibrasyon/OOD atlandı: {e}")
        return None
    p_te = kal.olasilik(s_te)
    naif = (s_te - s_te.min()) / (s_te.max() - s_te.min() + 1e-12)
    ece_ham, ece_kalibre = MT.ece(y_test, naif), MT.ece(y_test, p_te)

    s_normal_te = s_te[y_test == 0]
    ood_gurultu = OOD.ood_karsilastir(s_normal_te, model.anomali_skoru(OOD.gurultu_uret(Xte, seed=SEED)))
    ood_karisik = OOD.ood_karsilastir(s_normal_te, model.anomali_skoru(OOD.karistir_uret(Xte, seed=SEED)))
    print(
        f"    kalibrasyon: ECE {ece_ham:.3f}->{ece_kalibre:.3f}  |  "
        f"OOD AUC gürültü {ood_gurultu['ayrim_auc']:.3f} / karışık {ood_karisik['ayrim_auc']:.3f}"
    )
    return {
        "model": ad, "ece_ham": ece_ham, "ece_kalibre": ece_kalibre,
        "ood_gurultu_auc": ood_gurultu["ayrim_auc"], "ood_karisik_auc": ood_karisik["ayrim_auc"],
    }


def oznitelik_kismi(strateji: str, hizli: bool) -> tuple[list[dict], dict | None]:
    veri = OZ.yukle(strateji)
    print(f"  öznitelik  {veri.ozet()}")
    olc = Olcekleyici().egit(veri.X_train_normal)
    Xtr = olc.uygula(veri.X_train_normal)
    Xva, Xte = olc.uygula(veri.X_val), olc.uygula(veri.X_test)
    Xva_n = Xva[veri.y_val == 0] if (veri.y_val == 0).any() else None
    saat = float((veri.y_test == 0).sum()) * config.PENCERE_SN / 3600.0
    kir = veri.kirilim("makine_tipi")

    isler = [
        (B.RMSEsigi(veri.oznitelik_adlari), "ham"),
        (B.IzolasyonOrmani(seed=SEED), "olcek"),
        (B.Mahalanobis(), "olcek"),
    ]
    if TORCH_VAR:
        isler.append((DenseAE(epok=8 if hizli else 120, seed=SEED), "ae"))

    satirlar = []
    kalib_ood = None
    for model, tur in isler:
        t0 = time.perf_counter()
        try:
            if tur == "ham":
                model.egit(veri.X_train_normal)
                s_va, s_te = model.anomali_skoru(veri.X_val), model.anomali_skoru(veri.X_test)
            elif tur == "ae":
                model.egit(Xtr, X_val=Xva_n)
                s_va, s_te = model.anomali_skoru(Xva), model.anomali_skoru(Xte)
            else:
                model.egit(Xtr)
                s_va, s_te = model.anomali_skoru(Xva), model.anomali_skoru(Xte)
        except ValueError as e:
            print(f"    {model.ad:20s} atlandı: {e}")
            continue
        s = _degerlendir(model.ad, veri.y_test, s_te, veri.y_val, s_va, kir, saat, time.perf_counter() - t0)
        if s:
            satirlar.append(s)
        if tur == "ae":  # yalnız DenseAE için — 7 modelin hepsinde tekrarlamaya gerek yok
            kalib_ood = _kalibrasyon_ood_calistir(model.ad, model, Xte, veri.y_test, Xva, veri.y_val, s_te, s_va)
    return satirlar, kalib_ood


def _mel_olcekle(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu[None, :, None]) / sd[None, :, None]


def mel_kismi(strateji: str, hizli: bool) -> list[dict]:
    if not TORCH_VAR or not (config.ISLENMIS_VERI / "mel.npz").exists():
        return []
    veri = MEL.yukle(strateji)
    print(f"  mel        {veri.ozet()}")
    Xn = veri.X_train_normal
    mu = Xn.mean(axis=(0, 2))
    sd = Xn.std(axis=(0, 2))
    sd[sd < 1e-6] = 1.0
    Xtr = _mel_olcekle(Xn, mu, sd)
    Xva, Xte = _mel_olcekle(veri.X_val, mu, sd), _mel_olcekle(veri.X_test, mu, sd)
    Xva_n = Xva[veri.y_val == 0] if (veri.y_val == 0).any() else None
    saat = float((veri.y_test == 0).sum()) * config.PENCERE_SN / 3600.0
    kir = veri.kirilim("makine_tipi")

    e = 6 if hizli else 100
    modeller = [
        ConvAE(epok=e, seed=SEED),
        CRNNAE(epok=e, seed=SEED),
        SpekTransformerAE(epok=e, seed=SEED),
    ]
    satirlar = []
    for model in modeller:
        t0 = time.perf_counter()
        try:
            model.egit(Xtr, X_val=Xva_n)
            s_va, s_te = model.anomali_skoru(Xva), model.anomali_skoru(Xte)
        except ValueError as ex:
            print(f"    {model.ad:20s} atlandı: {ex}")
            continue
        s = _degerlendir(model.ad, veri.y_test, s_te, veri.y_val, s_va, kir, saat, time.perf_counter() - t0)
        if s:
            satirlar.append(s)
    return satirlar


def fuzyon_kismi(strateji: str, hizli: bool) -> list[dict]:
    if not TORCH_VAR or not (config.ISLENMIS_VERI / "mel.npz").exists():
        return []
    mel_veri = MEL.yukle(strateji)
    tel_veri = OZ.yukle(strateji, gruplar=("t",))
    hizali = (
        np.array_equal(mel_veri.y_train, tel_veri.y_train)
        and np.array_equal(mel_veri.y_val, tel_veri.y_val)
        and np.array_equal(mel_veri.y_test, tel_veri.y_test)
    )
    if not hizali:
        print("    fusion_ae            atlandı: mel ve öznitelik satırları hizasız "
              "(02_oznitelik_cikar.py --mel'i birlikte çalıştır)")
        return []

    print(f"  füzyon     mel {mel_veri.sekil} ⊕ telemetri {tel_veri.X_train.shape[1]}d")
    mu = mel_veri.X_train_normal.mean(axis=(0, 2))
    sd = mel_veri.X_train_normal.std(axis=(0, 2))
    sd[sd < 1e-6] = 1.0
    Xs_tr = _mel_olcekle(mel_veri.X_train_normal, mu, sd)
    Xs_va, Xs_te = _mel_olcekle(mel_veri.X_val, mu, sd), _mel_olcekle(mel_veri.X_test, mu, sd)

    olc = Olcekleyici().egit(tel_veri.X_train_normal)
    Xt_tr = olc.uygula(tel_veri.X_train_normal)
    Xt_va, Xt_te = olc.uygula(tel_veri.X_val), olc.uygula(tel_veri.X_test)

    normal_va = tel_veri.y_val == 0
    Xs_va_n = Xs_va[normal_va] if normal_va.any() else None
    Xt_va_n = Xt_va[normal_va] if normal_va.any() else None
    saat = float((tel_veri.y_test == 0).sum()) * config.PENCERE_SN / 3600.0
    kir = tel_veri.kirilim("makine_tipi")

    model = FusionAE(epok=8 if hizli else 100, seed=SEED)
    t0 = time.perf_counter()
    try:
        model.egit(Xs_tr, Xt_tr, X_spek_val=Xs_va_n, X_tel_val=Xt_va_n)
        s_va = model.anomali_skoru(Xs_va, Xt_va)
        s_te = model.anomali_skoru(Xs_te, Xt_te)
    except ValueError as e:
        print(f"    {model.ad:20s} atlandı: {e}")
        return []
    satir = _degerlendir(model.ad, tel_veri.y_test, s_te, tel_veri.y_val, s_va, kir, saat, time.perf_counter() - t0)
    return [satir] if satir else []


def strateji_calistir(strateji: str, hizli: bool) -> tuple[pd.DataFrame, dict | None]:
    oz_satirlar, kalib_ood = oznitelik_kismi(strateji, hizli)
    satirlar = oz_satirlar + mel_kismi(strateji, hizli) + fuzyon_kismi(strateji, hizli)
    if not satirlar:
        raise ValueError("hiçbir model değerlendirilemedi (tek sınıflı test/val?)")
    df = pd.DataFrame(satirlar)
    df.insert(0, "strateji", strateji)
    if kalib_ood:
        kalib_ood = {**kalib_ood, "strateji": strateji}
    return df, kalib_ood


def _gercek_mimii_mi() -> bool:
    kok = config.HAM_VERI / "mimii"
    return kok.exists() and any(kok.rglob("*.wav"))


def rapor_yaz(tablolar: list[pd.DataFrame], hizli: bool, kalib_ood_listesi: list[dict]) -> Path:
    config.SONUC_DIZINI.mkdir(parents=True, exist_ok=True)
    yol = config.SONUC_DIZINI / "faz2-modeller.md"
    goster = ["strateji", "model", "AUC", "AUC_CI", "pAUC", "F1", "precision", "recall", "FPR", "yanlış_alarm/sa", "sn"]
    mel_var = (config.ISLENMIS_VERI / "mel.npz").exists()
    veri_notu = (
        "- veri: GERÇEK MIMII (Zenodo 3384388) — sayılar savunulabilir, portföy değeri var"
        if _gercek_mimii_mi()
        else "- NOT: sentetik veri — sayılar boru hattı doğrulaması içindir, "
        "portföy değeri için MIMII gerekir"
    )
    parcalar = [
        "# Faz 2 — model karşılaştırması\n",
        "- öznitelik yolu: a_ + z_ + t_  ·  mel yolu: log-mel tensör "
        f"({'aktif' if mel_var and TORCH_VAR else 'yok'})",
        f"- torch: {'var' if TORCH_VAR else 'YOK — yalnız baseline'}"
        f"{'  (hızlı mod)' if hizli and TORCH_VAR else ''}",
        "- gözetimsiz (yalnız normal eğitim); eşik validasyondan F1 ile",
        f"{veri_notu}\n",
    ]
    tum = pd.concat(tablolar, ignore_index=True)
    parcalar.append(tum[goster].round(3).to_markdown(index=False))

    parcalar.append("\n## Makine tipine göre AUC kırılımı\n")
    for _, satir in tum.iterrows():
        k = satir["_kirilim"]
        if k:
            s = ", ".join(f"{g}: {v['auc']:.3f} (n={v['n']})" for g, v in k.items())
            parcalar.append(f"- **{satir['strateji']} / {satir['model']}** — {s}")

    if {"birim_disi", "rejim_holdout"} <= set(tum["strateji"]):
        parcalar.append("\n## Domain-shift (RB-01)\n")
        for model in tum["model"].unique():
            bd = tum[(tum.strateji == "birim_disi") & (tum.model == model)]["AUC"]
            rh = tum[(tum.strateji == "rejim_holdout") & (tum.model == model)]["AUC"]
            if len(bd) and len(rh):
                parcalar.append(
                    f"- {model}: birim_disi {bd.iloc[0]:.3f} → rejim_holdout {rh.iloc[0]:.3f} "
                    f" (Δ {float(bd.iloc[0] - rh.iloc[0]):+.3f})"
                )

    if kalib_ood_listesi:
        parcalar.append(
            "\n## Kalibrasyon + OOD (dense_ae) — RB benzeri sağlamlık kontrolü\n\n"
            "ECE: ham (min-max normalize skor) vs Platt-kalibre olasılık, validasyon üzerinde "
            "uydurulmuş, test'te ölçülmüş. OOD AUC: eğitilmiş model, iç-dağılım normali "
            "gürültü/öznitelik-karışık (yapısı bozulmuş) girdiden ayırt edebiliyor mu "
            "(1.0 = tam ayırt ediyor, 0.5 = ayıramıyor → 'kendinden emin yanlış' riski).\n"
        )
        parcalar.append("| strateji | model | ECE ham | ECE kalibre | OOD AUC (gürültü) | OOD AUC (karışık) |")
        parcalar.append("|---|---|---:|---:|---:|---:|")
        for k in kalib_ood_listesi:
            parcalar.append(
                f"| {k['strateji']} | {k['model']} | {k['ece_ham']:.3f} | {k['ece_kalibre']:.3f} | "
                f"{k['ood_gurultu_auc']:.3f} | {k['ood_karisik_auc']:.3f} |"
            )

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
    print(f"torch: {'var' if TORCH_VAR else 'YOK'}  ·  mel.npz: {(config.ISLENMIS_VERI / 'mel.npz').exists()}")
    tablolar = []
    kalib_ood_listesi = []
    for s in stratejiler:
        print(f"\n[{s}]")
        try:
            df, kalib_ood = strateji_calistir(s, a.hizli)
            tablolar.append(df)
            if kalib_ood:
                kalib_ood_listesi.append(kalib_ood)
        except (ValueError, FileNotFoundError) as e:
            print(f"  atlandı: {e}")

    if not tablolar:
        print("HATA: hiçbir strateji çalışmadı")
        return 1
    yol = rapor_yaz(tablolar, a.hizli, kalib_ood_listesi)
    print(f"\nyazildi -> {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
