"""Faz 2 — SQL tabanlı RUL hattı: gerçek C-MAPSS -> PostgreSQL -> SQL öznitelikleri -> model -> risk.

1. train/test_FD001..4 ölçümleri `cmapss_olcum` tablosuna yüklenir (hacim raporlanır)
2. Eğitim öznitelikleri SQL pencere fonksiyonlarıyla üretilir; sorgu süresi ölçülür
   ve sonuç bağımsız pandas uygulamasıyla karşılaştırılır (parite)
3. FD001 train motorları -> train/val (motor bazında); model seçimi YALNIZ val'de
4. Seçilen yapılandırma tüm train motorlarıyla yeniden eğitilir; resmi FD001 testi
   (100 motor, motor başına son çevrim, RUL_FD001.txt) BİR KEZ değerlendirilir
5. Risk sıralaması + sıralama metrikleri; model artefaktı API için kaydedilir

Çıktı:  sonuclar/faz2-rul-sql.md, modeller/rul/hgb_fd001.joblib

Kullanım:
  docker compose up -d postgres
  set NABIZ_PG_DSN=postgresql://nabiz:nabiz_yerel@localhost:55432/nabiz
  python 09_rul_sql.py
  python 09_rul_sql.py --dsn sqlite:///data/cmapss.db    # PostgreSQL'siz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import cmapss as C  # noqa: E402
from nabiz.data import sql_oznitelik as S  # noqa: E402
from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.models import rul_risk as R  # noqa: E402

utf8_cikti()

SEED = config.VARSAYILAN_SEED
ALT_KUMELER = ("FD001", "FD002", "FD003", "FD004")
HEDEF = "FD001"
# Küçük, önceden sabitlenmiş aday kümesi — seçim yalnız val motorlarında.
ADAYLAR = {
    "hgb_sig": dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40),
    "hgb_orta": dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20),
    "hgb_derin": dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=20),
}


def rul_etiketi(df: pd.DataFrame) -> np.ndarray:
    """Run-to-failure train satırı için kırpılmış RUL: min(max_cevrim - cevrim, 125)."""
    maks = df.groupby("birim")["cevrim"].transform("max")
    return np.minimum(maks - df["cevrim"], C.TAVAN_RUL).to_numpy(dtype=float)


def metrikler(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
        "MAE": float(np.mean(np.abs(p - y))),
        "cmapss": MT.cmapss_skoru(y, p),
    }


def explain_suresi_ms(b: S.Baglanti, sensorler: list[str]) -> float | None:
    if b.lehce != "postgresql":
        return None
    cur = b.con.cursor()
    cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + S.oznitelik_sorgusu(sensorler, b.isaret, kaynak_filtresi=False))
    return float(cur.fetchone()[0][0]["Execution Time"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=os.environ.get("NABIZ_PG_DSN"),
                    help="postgresql://... veya sqlite:///yol.db (öntanım: NABIZ_PG_DSN)")
    ap.add_argument("--yukleme-atla", action="store_true", help="veri zaten DB'de")
    a = ap.parse_args()
    if not a.dsn:
        print("HATA: --dsn ya da NABIZ_PG_DSN gerekli (bkz. docstring)")
        return 1

    kok = config.HAM_VERI / "cmapss"
    if not (kok / f"train_{HEDEF}.txt").exists():
        print(f"HATA: {kok}/train_{HEDEF}.txt yok — python 00_veri_indir.py --cmapss-bilgi")
        return 1

    b = S.baglan(a.dsn)
    S.sema_olustur(b)
    print(f"DB: {b.lehce}")

    # 1) yükleme + hacim
    if not a.yukleme_atla:
        for ad in ALT_KUMELER:
            t0 = time.perf_counter()
            n_tr = S.yukle(b, C.train_oku(kok / f"train_{ad}.txt"), f"{ad}-train")
            n_te = S.yukle(b, C.test_oku(kok / f"test_{ad}.txt", kok / f"RUL_{ad}.txt"), f"{ad}-test")
            print(f"  {ad}: train {n_tr} + test {n_te} satir yuklendi ({time.perf_counter() - t0:.1f} sn)")
    hacim = S.satir_sayilari(b)
    toplam = sum(hacim.values())
    print(f"toplam {toplam} olcum satiri, {len(hacim)} kaynak")

    # 2) sensör seçimi (YALNIZ train motorlarıyla) + SQL öznitelikleri
    ham_train = C.train_oku(kok / f"train_{HEDEF}.txt")
    atama = C.birim_split(ham_train["birim"].to_numpy(), test_orani=0.0, val_orani=0.2, seed=SEED)
    tr_birim = sorted(k for k, v in atama.items() if v == "train")
    va_birim = sorted(k for k, v in atama.items() if v == "val")
    sensorler = S.bilgilendirici_sensorler(ham_train[ham_train["birim"].isin(tr_birim)])
    print(f"secilen sensorler ({len(sensorler)}): {sensorler}")
    ozn = S.oznitelik_adlari(sensorler)

    oz_train = S.oznitelik_cek(b, sensorler, f"{HEDEF}-train")
    oz_test = S.oznitelik_cek(b, sensorler, f"{HEDEF}-test", yalniz_son_cevrim=True)
    rul_test = S.test_rul_son(b, f"{HEDEF}-test")

    # parite: SQL (bu DB) vs bağımsız pandas uygulaması
    ref = S.oznitelik_pandas(ham_train, sensorler)
    maks_fark = max(
        float(np.nanmax(np.abs(oz_train[c].to_numpy() - ref[c].to_numpy()) / (1.0 + np.abs(ref[c].to_numpy()))))
        for c in ozn[1:]
    )
    nan_ayni = all((oz_train[c].isna().to_numpy() == ref[c].isna().to_numpy()).all() for c in ozn[1:])
    print(f"parite SQL vs pandas: maks goreli fark {maks_fark:.2e}, NaN deseni ayni: {nan_ayni}")
    if maks_fark > 1e-6 or not nan_ayni:
        print("HATA: SQL ve pandas oznitelikleri uyusmuyor")
        return 1

    sure_hedef = S.sorgu_suresi(b, sensorler, f"{HEDEF}-train")
    sure_tum = S.sorgu_suresi(b, sensorler, None)
    explain_ms = explain_suresi_ms(b, sensorler)
    print(f"oznitelik sorgusu: {HEDEF}-train {sure_hedef:.2f} sn, tum {toplam} satir {sure_tum:.2f} sn"
          + (f" (EXPLAIN ANALYZE yurutme {explain_ms:.0f} ms)" if explain_ms else ""))

    # 3) model seçimi — yalnız val
    y_train_tum = rul_etiketi(oz_train)
    tr_m = oz_train["birim"].isin(tr_birim).to_numpy()
    # val'i resmi test protokolüne benzet: her val motorundan tek, rastgele bir
    # kesilme noktası (son çevrim değil — o nokta hep RUL=0 olurdu)
    rng = np.random.default_rng(SEED)
    va_idx = [
        int(rng.choice(g.index[30:] if len(g) > 60 else g.index))
        for _, g in oz_train[oz_train["birim"].isin(va_birim)].groupby("birim")
    ]
    X, y = oz_train[ozn].to_numpy(), y_train_tum
    val_sonuc = {"ort_taban": metrikler(y[va_idx], np.full(len(va_idx), y[tr_m].mean()))}
    for ad, par in ADAYLAR.items():
        m = HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **par).fit(X[tr_m], y[tr_m])
        val_sonuc[ad] = metrikler(y[va_idx], np.clip(m.predict(X[va_idx]), 0, None))
        print(f"  val {ad:10s} RMSE {val_sonuc[ad]['RMSE']:.2f}  MAE {val_sonuc[ad]['MAE']:.2f}")
    secilen = min(ADAYLAR, key=lambda k: val_sonuc[k]["RMSE"])
    print(f"secilen: {secilen} (val RMSE)")

    # 4) tüm train motorlarıyla yeniden eğit, resmi testte BİR KEZ değerlendir
    model = HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **ADAYLAR[secilen]).fit(X, y)
    X_te = oz_test[ozn].to_numpy()
    y_te = np.minimum(rul_test.loc[oz_test["birim"]].to_numpy(), C.TAVAN_RUL)
    p_te = np.clip(model.predict(X_te), 0, None)
    test_m = metrikler(y_te, p_te)
    taban_m = metrikler(y_te, np.full_like(y_te, y.mean()))
    # bazı çalışmalar test RUL'unu kırpmaz — ikisi de raporlanır
    y_te_ham = rul_test.loc[oz_test["birim"]].to_numpy()
    test_ham_m = metrikler(y_te_ham, p_te)

    # sızıntı kontrolleri: (a) etiketler karıştırılınca taban düzeyine çökmeli,
    # (b) yalnız `cevrim` (sensörsüz) ne verir — kazanım sensörlerden mi geliyor
    par = ADAYLAR[secilen]
    y_kar = np.random.default_rng(SEED).permutation(y)
    kar_rmse = metrikler(y_te, HistGradientBoostingRegressor(
        random_state=SEED, early_stopping=False, **par).fit(X, y_kar).predict(X_te))["RMSE"]
    cev_rmse = metrikler(y_te, HistGradientBoostingRegressor(
        random_state=SEED, early_stopping=False, **par).fit(oz_train[["cevrim"]].to_numpy(), y)
        .predict(oz_test[["cevrim"]].to_numpy()))["RMSE"]
    print(f"sizinti kontrolu: karistirilmis etiket RMSE {kar_rmse:.2f}, yalniz cevrim RMSE {cev_rmse:.2f}")
    print(f"[resmi test, {len(y_te)} motor] {secilen}: RMSE {test_m['RMSE']:.2f}  MAE {test_m['MAE']:.2f}  "
          f"C-MAPSS {test_m['cmapss']:.1f}  |  ort_taban RMSE {taban_m['RMSE']:.2f}")

    # 5) risk
    sirali = R.risk_sirala(oz_test["birim"].to_numpy(), p_te)
    sirali["gercek_rul"] = rul_test.loc[sirali["birim"]].to_numpy()
    sm = R.siralama_metrikleri(y_te, p_te)
    kar = R.seviye_karisiklik(y_te, p_te)
    print(f"siralama: spearman {sm['spearman']:.3f}, kritik yakalama {sm['kritik_yakalama']:.2f} "
          f"({sm['n_kritik']} kritik motor), ust-{sm['ust_k']} kesinlik {sm['kritik_kesinlik_ust_k']:.2f}")

    model_yolu = config.MODEL_DIZINI / "rul" / f"hgb_{HEDEF.lower()}.joblib"
    model_yolu.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "sensorler": sensorler, "oznitelikler": ozn, "yapilandirma": secilen,
                 "hedef": HEDEF, "test": test_m}, model_yolu)

    # rapor
    satirlar = [
        "# Faz 2 — SQL tabanlı RUL hattı (gerçek C-MAPSS)\n",
        f"- veri: NASA C-MAPSS (PCoE), `{HEDEF}` modellendi; DB: **{b.lehce}**",
        f"- hacim: **{toplam:,}** ölçüm satırı, {len(hacim)} kaynak (FD001–FD004 train+test)",
        f"- sensörler (train motorlarından, std > 0.01): {len(sensorler)} — "
        f"{', '.join(s.replace('sensor_', 's') for s in sensorler)}",
        f"- öznitelik: {len(ozn)} (SQL pencere fonksiyonları: ort10/30, eğim10/30, var30, sapma)",
        f"- bölme: {len(tr_birim)} train / {len(va_birim)} val motor (motor bazında); "
        f"resmi test: {len(y_te)} motor, motor başına son çevrim\n",
        "## Hacim ve sorgu performansı\n",
        "| kaynak | satır |", "|---|---:|",
        *[f"| {k} | {v:,} |" for k, v in hacim.items()],
        "",
        f"- öznitelik sorgusu, `{HEDEF}-train` ({hacim.get(f'{HEDEF}-train', 0):,} satır): {sure_hedef:.2f} sn "
        "(satırlar Python'a çekilerek, 3 denemenin en iyisi)",
        f"- tüm tablo ({toplam:,} satır): {sure_tum:.2f} sn"
        + (f"; `EXPLAIN ANALYZE` sunucu yürütme süresi {explain_ms:.0f} ms" if explain_ms else ""),
        f"- parite (SQL vs bağımsız pandas, {len(oz_train):,} satır × {len(ozn) - 1} öznitelik): "
        f"maks göreli fark {maks_fark:.1e}, NaN deseni aynı\n",
        "NOT: Bu hacim küçük-orta ölçektir (tek makine, ~265 bin satır); \"büyük veri\" iddiası "
        "değildir. Sayı olduğu gibi raporlanır, veri çoğaltılmamıştır.\n",
        "## Model seçimi (yalnız val motorları, rastgele kesilme noktası)\n",
        "| model | RMSE | MAE |", "|---|---:|---:|",
        *[f"| {k}{' ← seçildi' if k == secilen else ''} | {v['RMSE']:.2f} | {v['MAE']:.2f} |"
          for k, v in val_sonuc.items()],
        "",
        f"## Resmi FD001 testi ({len(y_te)} motor, bir kez)\n",
        "| model | RMSE | MAE | C-MAPSS skoru (düşük iyi) |", "|---|---:|---:|---:|",
        f"| {secilen} (SQL öznitelikleri) | {test_m['RMSE']:.2f} | {test_m['MAE']:.2f} | {test_m['cmapss']:.1f} |",
        f"| ort_taban (train ortalaması) | {taban_m['RMSE']:.2f} | {taban_m['MAE']:.2f} | {taban_m['cmapss']:.1f} |",
        "",
        f"Test RUL'u 125'te kırpılmıştır (literatürde yaygın). Kırpılmamış gerçek RUL'a karşı: "
        f"RMSE {test_ham_m['RMSE']:.2f}, MAE {test_ham_m['MAE']:.2f} "
        f"({int((y_te_ham > C.TAVAN_RUL).sum())} motorun gerçek RUL'u 125'in üstünde).\n",
        "Sızıntı kontrolleri: eğitim etiketleri karıştırılınca test RMSE "
        f"**{kar_rmse:.2f}** (taban {taban_m['RMSE']:.2f}); sensörsüz, yalnız `cevrim` ile "
        f"**{cev_rmse:.2f}** — kazanım sensör özniteliklerinden geliyor. Test öznitelikleri yalnız "
        "kesik test kayıtlarından, nedensel pencerelerle hesaplanır (bkz. `tests/test_sql_oznitelik.py`).\n",
        "## Bakım önceliklendirme\n",
        "Risk eşikleri (tahmini RUL, çevrim): " + ", ".join(f"{s} ≤ {e:g}" for e, s in R.RISK_ESIKLERI)
        + ", aksi NORMAL — planlama varsayımı, veri gereksinimi değil.\n",
        f"- Spearman (tahmin vs gerçek RUL sırası): **{sm['spearman']:.3f}**",
        f"- gerçek RUL ≤ 30 olan {sm['n_kritik']} motorun KRITIK/YUKSEK işaretlenme oranı: "
        f"**{sm['kritik_yakalama']:.0%}**",
        f"- en riskli {sm['ust_k']} tahminden gerçekten kritik (RUL ≤ 30) olan: "
        f"**{sm['kritik_kesinlik_ust_k']:.0%}**\n",
        "Seviye karışıklığı (satır gerçek, sütun tahmin):\n",
        "| gerçek \\ tahmin | " + " | ".join(kar.columns) + " |",
        "|---|" + "---:|" * len(kar.columns),
        *[f"| {i} | " + " | ".join(str(int(v)) for v in kar.loc[i]) + " |" for i in kar.index],
        "",
        "İlk 10 (en riskli):\n",
        "| sıra | motor | tahmini RUL | risk | gerçek RUL |", "|---:|---:|---:|---|---:|",
        *[f"| {r.sira} | {r.birim} | {r.tahmini_rul:.1f} | {r.risk} | {r.gercek_rul:.0f} |"
          for r in sirali.head(10).itertuples()],
        "",
        "## Kapsam ve sınırlar\n",
        "- C-MAPSS **simüle edilmiş** motor verisidir (NASA); gerçek filo verisi değildir. "
        "Sonuçlar bir havacılık bakım senaryosu prototipidir.",
        "- Yalnız FD001 (tek çalışma koşulu, tek arıza modu) modellendi; FD002/FD004 çoklu "
        "çalışma koşulu için koşul-bazlı normalizasyon gerektirir (yapılmadı).",
        "- Seçim yalnız val'de yapıldı; test sonucuna bakılarak hiçbir ayar değiştirilmedi. "
        "Tek tohum — belirsizlik aralığı raporlanmadı.",
        "- Aynı resmi testte ham 21 sensör penceresiyle GRU: `sonuclar/faz2-rul.md`.",
    ]
    yol = config.SONUC_DIZINI / "faz2-rul-sql.md"
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print(f"\nyazildi -> {yol}\nmodel -> {model_yolu}")
    b.kapat()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
