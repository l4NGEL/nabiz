"""Faz 4 — C++ çıkarım yolu için golden-vector üretimi (Python↔C++ parite).

`04_uc_uyarla.py`'nin ürettiği ONNX modelini (modeller/edge/dense_ae_fp32.onnx)
Python ONNX Runtime ile çalıştırır; test kümesinden N örnek + skorları
cpp/testler/ altına yazar. `cpp/testler/dogrula.py` bunları derlenmiş C++
ikilisinin çıktısıyla karşılaştırır — aynı .onnx dosyası, iki farklı dil
binding'i (Python vs C++), aynı sayı mı üretiyor sorusunun cevabı.

Kullanım:
  python 06_golden_uret.py                 # önce: python 04_uc_uyarla.py
  python 06_golden_uret.py --n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import oznitelik as OZ  # noqa: E402
from nabiz.edge.onnx_disa import yeniden_yapilandirma_skoru  # noqa: E402
from nabiz.models.temel import Olcekleyici  # noqa: E402

utf8_cikti()

CPP_TEST_DIZINI = Path(__file__).parent / "cpp" / "testler"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strateji", default="birim_disi")
    ap.add_argument("--n", type=int, default=40, help="golden örnek sayısı")
    ap.add_argument("--model", default=None, help="öntanım: modeller/edge/dense_ae_fp32.onnx")
    a = ap.parse_args()

    model_yolu = Path(a.model) if a.model else config.MODEL_DIZINI / "edge" / "dense_ae_fp32.onnx"
    if not model_yolu.exists():
        print(f"HATA: {model_yolu} yok. Önce:  python 04_uc_uyarla.py")
        return 1

    veri = OZ.yukle(a.strateji)
    olc = Olcekleyici().egit(veri.X_train_normal)
    Xte = olc.uygula(veri.X_test).astype(np.float32)
    n = min(a.n, len(Xte))
    rng = np.random.default_rng(config.VARSAYILAN_SEED)
    idx = rng.choice(len(Xte), size=n, replace=False)
    X = Xte[idx]

    oturum = ort.InferenceSession(str(model_yolu), providers=["CPUExecutionProvider"])
    ad = oturum.get_inputs()[0].name
    yeniden = oturum.run(None, {ad: X})[0]
    skor = yeniden_yapilandirma_skoru(X, yeniden)

    CPP_TEST_DIZINI.mkdir(parents=True, exist_ok=True)
    np.savetxt(CPP_TEST_DIZINI / "altin_girdi.csv", X, delimiter=",", fmt="%.8e")
    np.savetxt(CPP_TEST_DIZINI / "altin_skor.csv", skor, delimiter=",", fmt="%.8e")

    print(f"model: {model_yolu.name}  ·  {n} örnek x {X.shape[1]} öznitelik")
    print(f"skor aralığı: [{skor.min():.4f}, {skor.max():.4f}]")
    print(f"yazildi -> {CPP_TEST_DIZINI / 'altin_girdi.csv'}")
    print(f"yazildi -> {CPP_TEST_DIZINI / 'altin_skor.csv'}")
    print(f"\nsonra:  python cpp/testler/dogrula.py --model {model_yolu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
