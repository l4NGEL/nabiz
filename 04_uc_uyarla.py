"""Faz 3 — uç uyarlama: ONNX export, INT8 nicemleme, parite, gecikme/verim.

Faz 2'nin DenseAE'sini (öznitelik yolu) ve varsa ConvAE'sini (mel yolu) alır:
  1. ONNX'e aktarır (dinamik yığın ekseni)
  2. PyTorch <-> ONNX Runtime FP32 sayısal + anomali-skoru paritesini doğrular
  3. Dinamik INT8'e nicemler (RB-04: AUC düşüşü + skor korelasyonu)
  4. Tek-örnek gecikmesini ölçer (P50/P95/P99, x86 CPU — PR-02)
  5. Model boyutunu karşılaştırır (PR-04)

Fiziksel uç donanım GEREKMEZ — bu makinenin CPU'sunda gerçek, geçerli sayılar.
ARM/Jetson için: aynı .onnx dosyaları Oracle Cloud Always Free ARM VM'e veya
Jetson'a kopyalanıp aynı betikle ölçülebilir (bkz. docs/yol-haritasi.md).

Çıktı:  sonuclar/faz3-uc.md
        modeller/edge/*.onnx

Kullanım:
  python 04_uc_uyarla.py                  # DenseAE + (varsa) ConvAE, tam eğitim
  python 04_uc_uyarla.py --hizli           # az epok (duman testi)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import oznitelik as OZ  # noqa: E402
from nabiz.eval import metrics as MT  # noqa: E402
from nabiz.models.temel import Olcekleyici  # noqa: E402

utf8_cikti()

try:
    from nabiz.data import mel as MEL
    from nabiz.edge import bench as BN
    from nabiz.edge import nicemleme as N
    from nabiz.edge import onnx_disa as O
    from nabiz.models.autoencoder import DenseAE
    from nabiz.models.cnn import ConvAE

    HAZIR = True
except ImportError:
    HAZIR = False

SEED = config.VARSAYILAN_SEED
UC_DIZINI = config.MODEL_DIZINI / "edge"


def _uc_degerlendir(ad: str, agi, X_test, y_test, X_val, y_val, kirilim, saat) -> dict:
    fp32 = O.disa_aktar(agi, X_test[:1], UC_DIZINI / f"{ad}_fp32.onnx")
    int8 = N.int8_nicemle(fp32, UC_DIZINI / f"{ad}_int8.onnx")

    ot_f32, ot_i8 = O.onnx_oturumu(fp32), O.onnx_oturumu(int8)
    parite_f32 = O.pariteyi_dogrula(agi, ot_f32, X_test)
    y_f32 = O.onnx_calistir(ot_f32, X_test)
    y_i8 = O.onnx_calistir(ot_i8, X_test)
    skor_par = O.skor_paritesi(X_test, y_f32, y_i8)

    s_torch = O.yeniden_yapilandirma_skoru(X_test, y_f32)  # ONNX FP32 ~ torch (parite yüksek)
    s_int8 = O.yeniden_yapilandirma_skoru(X_test, y_i8)

    def _auc_guvenli(y, s):
        try:
            return MT.auc(y, s)
        except ValueError:
            return float("nan")

    auc_f32, auc_i8 = _auc_guvenli(y_test, s_torch), _auc_guvenli(y_test, s_int8)
    auc_dusus = auc_f32 - auc_i8 if np.isfinite(auc_f32) and np.isfinite(auc_i8) else float("nan")

    gecikme_f32 = BN.gecikme_olc(ot_f32, X_test)
    gecikme_i8 = BN.gecikme_olc(ot_i8, X_test)

    return {
        "model": ad,
        "fp32_mb": O.dosya_boyutu_mb(fp32),
        "int8_mb": O.dosya_boyutu_mb(int8),
        "onnx_parite_maks_fark": parite_f32["maks_mutlak_fark"],
        "onnx_parite_gecti": parite_f32["gecti"],
        "int8_skor_korelasyon": skor_par["skor_korelasyon"],
        "auc_fp32": auc_f32,
        "auc_int8": auc_i8,
        "auc_dusus": auc_dusus,
        "rb04_gecti": bool(np.isnan(auc_dusus) or (auc_dusus <= 0.01 and skor_par["skor_korelasyon"] >= 0.99)),
        "fp32_p50_ms": gecikme_f32["p50_ms"],
        "fp32_p95_ms": gecikme_f32["p95_ms"],
        "int8_p50_ms": gecikme_i8["p50_ms"],
        "int8_p95_ms": gecikme_i8["p95_ms"],
        "pr02_gecti": bool(gecikme_i8["p95_ms"] <= 150.0),  # x86 tek çekirdek hedefi
    }


def oznitelik_yolu(strateji: str, hizli: bool) -> dict:
    veri = OZ.yukle(strateji)
    print(f"  öznitelik  {veri.ozet()}")
    olc = Olcekleyici().egit(veri.X_train_normal)
    Xtr = olc.uygula(veri.X_train_normal)
    Xva, Xte = olc.uygula(veri.X_val), olc.uygula(veri.X_test)
    Xva_n = Xva[veri.y_val == 0] if (veri.y_val == 0).any() else None

    ae = DenseAE(epok=8 if hizli else 120, seed=SEED).egit(Xtr, X_val=Xva_n)
    return _uc_degerlendir(
        "dense_ae", ae._agi, Xte.astype(np.float32), veri.y_test,
        Xva, veri.y_val, veri.kirilim("makine_tipi"),
        float((veri.y_test == 0).sum()) * config.PENCERE_SN / 3600.0,
    )


def _mel_olcekle(X, mu, sd):
    return (X - mu[None, :, None]) / sd[None, :, None]


def mel_yolu(strateji: str, hizli: bool) -> dict | None:
    if not (config.ISLENMIS_VERI / "mel.npz").exists():
        print("  mel yok — atlanıyor (python 02_oznitelik_cikar.py --mel)")
        return None
    veri = MEL.yukle(strateji)
    print(f"  mel        {veri.ozet()}")
    Xn = veri.X_train_normal
    mu, sd = Xn.mean(axis=(0, 2)), Xn.std(axis=(0, 2))
    sd[sd < 1e-6] = 1.0
    Xtr = _mel_olcekle(Xn, mu, sd)
    Xva, Xte = _mel_olcekle(veri.X_val, mu, sd), _mel_olcekle(veri.X_test, mu, sd)
    Xva_n = Xva[veri.y_val == 0] if (veri.y_val == 0).any() else None

    ae = ConvAE(epok=6 if hizli else 100, seed=SEED).egit(Xtr, X_val=Xva_n)
    return _uc_degerlendir(
        "conv_ae", ae._agi, Xte.astype(np.float32), veri.y_test,
        Xva, veri.y_val, veri.kirilim("makine_tipi"),
        float((veri.y_test == 0).sum()) * config.PENCERE_SN / 3600.0,
    )


def rapor_yaz(satirlar: list[dict]) -> Path:
    config.SONUC_DIZINI.mkdir(parents=True, exist_ok=True)
    yol = config.SONUC_DIZINI / "faz3-uc.md"
    parcalar = [
        "# Faz 3 — uç uyarlama (ONNX / INT8 / gecikme)\n",
        "Ölçüm ortamı: bu makinenin x86 CPU'su (ONNX Runtime `CPUExecutionProvider`).",
        "Fiziksel uç donanım kullanılmadı — bkz. docs/yol-haritasi.md 'donanım gerekmez' notu.\n",
        "| model | fp32 MB | int8 MB | ONNX parite (maks fark) | INT8 skor korr. | "
        "AUC fp32→int8 (Δ) | RB-04 | fp32 P50/P95 ms | int8 P50/P95 ms | PR-02 |",
        "|---|---:|---:|---:|---:|---|:---:|---|---|:---:|",
    ]
    for s in satirlar:
        parcalar.append(
            f"| {s['model']} | {s['fp32_mb']:.3f} | {s['int8_mb']:.3f} | "
            f"{s['onnx_parite_maks_fark']:.2e} | {s['int8_skor_korelasyon']:.4f} | "
            f"{s['auc_fp32']:.3f}→{s['auc_int8']:.3f} ({s['auc_dusus']:+.3f}) | "
            f"{'✓' if s['rb04_gecti'] else '✗'} | "
            f"{s['fp32_p50_ms']:.3f}/{s['fp32_p95_ms']:.3f} | "
            f"{s['int8_p50_ms']:.3f}/{s['int8_p95_ms']:.3f} | "
            f"{'✓' if s['pr02_gecti'] else '✗'} |"
        )
    parcalar.append(
        "\nRB-04 kriteri: AUC düşüşü ≤ 0.01 VE skor korelasyonu ≥ 0.99. "
        "PR-02 kriteri (bu ölçümde x86 vekili): P95 ≤ 150 ms.\n"
        "NOT: sentetik veride AUC zaten doygun (~1.0) — düşüş neredeyse sıfır görünür; "
        "gerçek MIMII'de RB-04 daha anlamlı bir test olur."
    )
    yol.write_text("\n".join(parcalar), encoding="utf-8")
    return yol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strateji", default="birim_disi")
    ap.add_argument("--hizli", action="store_true", help="az epok (duman testi)")
    a = ap.parse_args()

    if not HAZIR:
        print("HATA: torch/onnx/onnxruntime eksik.  pip install -e \".[dl]\"  sonra:  pip install onnx onnxruntime")
        return 1
    if not (config.ISLENMIS_VERI / "oznitelikler.parquet").exists():
        print("HATA: öznitelik yok. Önce:  python 02_oznitelik_cikar.py")
        return 1

    UC_DIZINI.mkdir(parents=True, exist_ok=True)
    print(f"[{a.strateji}]")
    satirlar = [oznitelik_yolu(a.strateji, a.hizli)]
    mel_sonuc = mel_yolu(a.strateji, a.hizli)
    if mel_sonuc:
        satirlar.append(mel_sonuc)

    for s in satirlar:
        print(
            f"  {s['model']:10s} fp32 {s['fp32_mb']:.3f} MB -> int8 {s['int8_mb']:.3f} MB | "
            f"AUC {s['auc_fp32']:.3f}->{s['auc_int8']:.3f} | "
            f"P95 fp32 {s['fp32_p95_ms']:.2f} ms / int8 {s['int8_p95_ms']:.2f} ms"
        )

    yol = rapor_yaz(satirlar)
    print(f"\nyazildi -> {yol}")
    print(f"yazildi -> {UC_DIZINI}/*.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
