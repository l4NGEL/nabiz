"""Ürün noktası — tip başına model, ünite uyarlaması, ablasyon, yanlış-alarm bütçesi.

Havuzlanmış Mahalanobis (mevcut taban), makine tipi başına Mahalanobis ve
ünite medyan/MAD uyarlamalı tip modeli aynı splitlerde karşılaştırılır.
Eşik F1 ile değil, tipin kendi validasyonunda FPR bütçesiyle de seçilir
(PR-01: ≤ 1 yanlış alarm / 24 saat).

Çıktı: sonuclar/faz2-urun.md

Kullanım:
  python 08_urun_degerlendir.py
  python 08_urun_degerlendir.py --strateji birim_disi
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
from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.eval.alarm import YONTEMLER, butce_sureklilik, dosya_auc, dosyalara_indir  # noqa: E402
from nabiz.eval.urun import KURULUMLAR, degerlendir_diziler, skorla  # noqa: E402

utf8_cikti()

ABLASYON = (
    ("a+z+t", None),
    ("a", ("a",)),
    ("z", ("z",)),
    ("t", ("t",)),
    ("a+z", ("a", "z")),
)


def _dilim(X: np.ndarray, adlar: list[str], gruplar: tuple[str, ...] | None) -> np.ndarray:
    if gruplar is None:
        return X
    idx = OZ.grup_indeksleri(adlar, gruplar)
    return X[:, idx]


def _calistir(veri: OZ.BolmeVerisi, kurulum: str, gruplar: tuple[str, ...] | None) -> dict:
    adlar = veri.oznitelik_adlari
    return degerlendir_diziler(
        _dilim(veri.X_train, adlar, gruplar), veri.y_train,
        veri.meta_train["makine_tipi"].to_numpy(), veri.meta_train["makine_birimi"].to_numpy(),
        _dilim(veri.X_val, adlar, gruplar), veri.y_val,
        veri.meta_val["makine_tipi"].to_numpy(), veri.meta_val["makine_birimi"].to_numpy(),
        _dilim(veri.X_test, adlar, gruplar), veri.y_test,
        veri.meta_test["makine_tipi"].to_numpy(), veri.meta_test["makine_birimi"].to_numpy(),
        kurulum=kurulum,
        pencere_sn=config.PENCERE_SN,
    )


def _satir(strateji: str, grup: str, rapor: dict, sn: float) -> dict:
    f1 = rapor["isletme"].get("f1", {})
    pr = rapor["isletme"].get("pr01", {})
    fpr = rapor["isletme"].get("fpr_0.1pct", {})
    tip_auc = " / ".join(f"{t} {v['auc']:.3f}" for t, v in sorted(rapor["tipler"].items()))
    return {
        "strateji": strateji,
        "kurulum": rapor["kurulum"],
        "grup": grup,
        "makro_auc": rapor["makro_auc"],
        "havuz_auc": rapor["havuz_auc"],
        "makro_pauc": rapor["makro_pauc"],
        "tip_auc": tip_auc,
        "f1_recall": f1.get("recall", float("nan")),
        "f1_alarm_sa": f1.get("yanlis_alarm_saat", float("nan")),
        "fpr01_recall": fpr.get("recall", float("nan")),
        "fpr01_alarm_sa": fpr.get("yanlis_alarm_saat", float("nan")),
        "pr01_recall": pr.get("recall", float("nan")),
        "pr01_alarm_sa": pr.get("yanlis_alarm_saat", float("nan")),
        "sn": sn,
    }


def _alarm_strateji(veri: OZ.BolmeVerisi, strateji: str) -> list[dict]:
    """Tip uyarlamalı akustik skor → dosya kararı. Yöntem validasyon recall'u ile seçilir."""
    if "dosya" not in veri.meta_test.columns or "pencere_idx" not in veri.meta_test.columns:
        print("  alarm atlandı: parquet'te dosya/pencere_idx yok")
        return []
    satirlar = []
    for grup_ad, gruplar in (("a", ("a",)), ("a+z", ("a", "z"))):
        t0 = time.perf_counter()
        try:
            s_va, s_te = skorla(
                "tip_uyarlama",
                _dilim(veri.X_train, veri.oznitelik_adlari, gruplar), veri.y_train,
                veri.meta_train["makine_tipi"].to_numpy(), veri.meta_train["makine_birimi"].to_numpy(),
                _dilim(veri.X_val, veri.oznitelik_adlari, gruplar),
                veri.meta_val["makine_tipi"].to_numpy(), veri.meta_val["makine_birimi"].to_numpy(),
                _dilim(veri.X_test, veri.oznitelik_adlari, gruplar),
                veri.meta_test["makine_tipi"].to_numpy(), veri.meta_test["makine_birimi"].to_numpy(),
            )
        except ValueError as e:
            print(f"  alarm {grup_ad:6s} atlandı: {e}")
            continue
        aday = []
        for yontem in YONTEMLER:
            ortak = dict(
                yontem=yontem, pencere_sn=config.PENCERE_SN, hop_sn=config.BINDIRME_SN,
            )
            val = dosyalara_indir(
                s_va, veri.y_val, veri.meta_val["dosya"].to_numpy(),
                veri.meta_val["makine_tipi"].to_numpy(), veri.meta_val["pencere_idx"].to_numpy(),
                birim=veri.meta_val["makine_birimi"].to_numpy(), **ortak,
            )
            te = dosyalara_indir(
                s_te, veri.y_test, veri.meta_test["dosya"].to_numpy(),
                veri.meta_test["makine_tipi"].to_numpy(), veri.meta_test["pencere_idx"].to_numpy(),
                birim=veri.meta_test["makine_birimi"].to_numpy(), **ortak,
            )
            try:
                karar = butce_sureklilik(val, te, MT.PR01_ALARM_SAAT)
                auc = dosya_auc(te)
            except ValueError as e:
                print(f"  alarm {grup_ad:6s} {yontem:6s} atlandı: {e}")
                continue
            tip_auc = " / ".join(f"{t} {v:.3f}" for t, v in sorted(auc["tipler"].items()))
            aday.append({
                "strateji": strateji,
                "grup": grup_ad,
                "yontem": yontem,
                "k": karar["k"],
                "val_recall": karar["val_recall"],
                "recall": karar["recall"],
                "olay_fp": karar["olay_fp"],
                "yanlis_alarm_saat": karar["yanlis_alarm_saat"],
                "pr01": karar["pr01"],
                "normal_saat": karar["normal_saat"],
                "makro_auc": auc["makro_auc"],
                "havuz_auc": auc["havuz_auc"],
                "tip_auc": tip_auc,
                "sn": time.perf_counter() - t0,
            })
        if not aday:
            continue
        secilen = max(aday, key=lambda r: (r["val_recall"], r["yontem"] == "medyan"))
        for r in aday:
            r["secildi"] = r["yontem"] == secilen["yontem"]
            satirlar.append(r)
            isaret = "→" if r["secildi"] else " "
            print(
                f"  alarm {isaret} {grup_ad:6s} {r['yontem']:6s} k={r['k']} "
                f"dosya AUC {r['makro_auc']:.3f}  recall {r['recall']:.3f}  "
                f"olay/sa {r['yanlis_alarm_saat']:.4f}  PR-01 {'geçti' if r['pr01'] else 'kaldı'}"
            )
    return satirlar


def rapor_yaz(satirlar: list[dict], alarmlar: list[dict] | None = None) -> Path:
    config.SONUC_DIZINI.mkdir(parents=True, exist_ok=True)
    yol = config.SONUC_DIZINI / "faz2-urun.md"
    df = pd.DataFrame(satirlar)
    goster = [
        "strateji", "kurulum", "grup", "makro_auc", "havuz_auc", "makro_pauc",
        "f1_recall", "f1_alarm_sa", "fpr01_recall", "fpr01_alarm_sa",
        "pr01_recall", "pr01_alarm_sa",
    ]
    parcalar = [
        "# Faz 2 — ürün noktası\n",
        "Üç kurulum, aynı Mahalanobis çekirdeği:",
        "- **havuz** — tüm tipler tek model (önceki raporun tabanı)",
        "- **tip** — fan ve pompa ayrı ayrı eğitilir; eşik tipin kendi validasyonundan",
        "- **tip_uyarlama** — her ünite medyan/MAD ile kendi taban çizgisine oturtulur "
        "(hedef bölmede etiketsiz). Görülmemiş ünitenin 'normal'i filodan uzaksa "
        "skor onu arıza sanmasın diye.",
        "",
        "İşletme noktaları (tip başına eşik, hatalar toplanır):",
        "- `f1` — validasyonda F1",
        "- `fpr01` — FPR ≈ 0.001 (2 sn pencerede ≈ 1.8 yanlış alarm/saat)",
        "- `pr01` — PR-01 bütçesi: ≤ 1 yanlış alarm / 24 saat (FPR ≈ 2.3e-5)",
        "",
        "Telemetri (`t_`) sentetik vekildir; ablasyon onu akustikten ayırır.",
        "",
        "## Karşılaştırma\n",
        df[goster].round(3).to_markdown(index=False),
        "",
        "## Tip kırılımı\n",
    ]
    kirilim = df[df.kurulum.isin(["havuz", "tip_uyarlama"]) & df.grup.isin(["a", "a+z", "a+z+t"])]
    for _, s in kirilim.iterrows():
        parcalar.append(
            f"- **{s['strateji']} / {s['kurulum']} / {s['grup']}** — {s['tip_auc']} "
            f"(makro {s['makro_auc']:.3f})"
        )
    uyar = df[df.kurulum == "tip_uyarlama"]
    if len(uyar):
        akustik = uyar[uyar.grup == "a"]
        parcalar.append("\n## Okuma\n")
        parcalar.append(
            "Ünite uyarlaması, havuzlanmış modelin görülmemiş ünitede çökmesini "
            "(havuz AUC ~0.40) tip-içi AUC ~0.90 bandına taşır. Telemetri (`t_`) "
            "her iki stratejide de AUC ~0.50 — sentetik kanal sinyal taşımıyor, "
            "füzyona katılmamalı. Zarflı akustik (`a+z`) SNR kaymasında en iyi; "
            "görülmemiş ünitede salt akustik (`a`) biraz daha yüksek."
        )
        if len(akustik):
            for _, s in akustik.iterrows():
                parcalar.append(
                    f"- `{s['strateji']}` akustik + uyarlama: makro AUC {s['makro_auc']:.3f}, "
                    f"pAUC {s['makro_pauc']:.3f}, FPR≈0.001 iken recall {s['fpr01_recall']:.3f} "
                    f"({s['fpr01_alarm_sa']:.2f} yanlış alarm/saat), PR-01 recall {s['pr01_recall']:.3f}."
                )
        pr_max = float(uyar["pr01_recall"].max())
        pauc_max = float(uyar["makro_pauc"].max())
        parcalar.append(
            f"\nBu koşuda PR-01 (≤ 1 yanlış alarm / 24 saat) recall en fazla {pr_max:.3f}. "
            f"En yüksek makro pAUC {pauc_max:.3f} (hedef 0.85). Sıralama düzeldi; "
            "düşük-FPR kuyruğu bu bütçeyi taşımıyor. Uyarlama etiketsizdir; "
            "istatistik, skorlanan partinin kendi pencerelerinden (medyan/MAD) gelir — "
            "tek pencereyi filo prototipine vurmak değildir. Pencere başına eşik "
            "PR-01'i tutmaz; alarm kararı dosya düzeyindedir."
        )
    if alarmlar:
        adf = pd.DataFrame(alarmlar)
        parcalar.append("\n## Dosya alarmı (PR-01)\n")
        parcalar.append(
            "Karar birimi dosyadır (~10 sn). Skor, pencerelerin medyanı veya p90'ıdır. "
            "Alarm, aynı kayıt klasöründe k ardışık dosya eşiği geçince bir kez sayılır. "
            "Eşik ham skor değil, her ünitenin kendi dosya medyan/MAD'ine göre z'dir; "
            "böylece başka ünitede seçilen eşik görülmemiş makineye taşınır. k, validasyonda "
            "≤ 1 olay / 24 saat bütçesini tutanlar arasından recall'u yüksek olandır; "
            "recall k'ya bağlı olmadığından eşitlikte en uzun süreklilik seçilir."
        )
        goster_a = [
            "strateji", "grup", "yontem", "k", "secildi", "makro_auc", "recall",
            "olay_fp", "yanlis_alarm_saat", "normal_saat", "pr01", "val_recall",
        ]
        parcalar.append("")
        parcalar.append(adf[goster_a].round(4).to_markdown(index=False))
        parcalar.append("\nSeçilenler:\n")
        for _, s in adf[adf.secildi].iterrows():
            durum = "PR-01 geçti" if s["pr01"] else "PR-01 kaldı"
            parcalar.append(
                f"- **{s['strateji']} / {s['grup']} / {s['yontem']} / k={int(s['k'])}** — "
                f"dosya makro AUC {s['makro_auc']:.3f} ({s['tip_auc']}), recall {s['recall']:.3f}, "
                f"{int(s['olay_fp'])} olay / {s['normal_saat']:.2f} saat normal "
                f"({s['yanlis_alarm_saat']:.4f}/saat). {durum}."
            )
    yol.write_text("\n".join(parcalar) + "\n", encoding="utf-8")
    return yol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strateji", default=None, help="tek strateji (öntanım: birim_disi + rejim_holdout)")
    a = ap.parse_args()
    config.dizinleri_hazirla()
    if not (config.ISLENMIS_VERI / "oznitelikler.parquet").exists():
        print("HATA: öznitelik yok. Önce:  python 02_oznitelik_cikar.py")
        return 1

    stratejiler = [a.strateji] if a.strateji else ["birim_disi", "rejim_holdout"]
    satirlar: list[dict] = []
    alarmlar: list[dict] = []
    for strateji in stratejiler:
        print(f"\n[{strateji}] yükleniyor")
        veri = OZ.yukle(strateji)
        print(f"  {veri.ozet()}")
        alarmlar.extend(_alarm_strateji(veri, strateji))
        for kurulum in KURULUMLAR:
            for grup_ad, gruplar in ABLASYON:
                if kurulum == "havuz" and grup_ad != "a+z+t":
                    continue
                t0 = time.perf_counter()
                try:
                    rapor = _calistir(veri, kurulum, gruplar)
                except ValueError as e:
                    print(f"  {kurulum:14s} {grup_ad:6s} atlandı: {e}")
                    continue
                sn = time.perf_counter() - t0
                satirlar.append(_satir(strateji, grup_ad, rapor, sn))
                print(
                    f"  {kurulum:14s} {grup_ad:6s} makro {rapor['makro_auc']:.3f}  "
                    f"havuz {rapor['havuz_auc']:.3f}  ({sn:.1f}s)"
                )
        del veri

    if not satirlar:
        print("HATA: hiçbir kurulum çalışmadı")
        return 1
    yol = rapor_yaz(satirlar, alarmlar)
    print(f"\nyazildi -> {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
