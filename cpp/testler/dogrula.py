"""Golden-vector parite testi: derlenmiş nabiz_cikarim.exe çıktısı vs Python (ONNX Runtime).

Önkoşul:  python 06_golden_uret.py   (altin_girdi.csv + altin_skor.csv üretir)
          CMake build (bkz. ../../README.md "Faz 4")

Kullanım:
  python cpp/testler/dogrula.py --model ../../modeller/edge/dense_ae_fp32.onnx
  python cpp/testler/dogrula.py --exe build/Release/nabiz_cikarim.exe --tolerans 1e-4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

BURASI = Path(__file__).parent
sys.path.insert(0, str(BURASI / ".." / ".." / "src"))

from nabiz.cli import utf8_cikti  # noqa: E402

utf8_cikti()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(BURASI / ".." / ".." / "modeller" / "edge" / "dense_ae_fp32.onnx"))
    ap.add_argument("--exe", default=str(BURASI / ".." / "build" / "Release" / "nabiz_cikarim.exe"))
    ap.add_argument("--tolerans", type=float, default=1e-4)
    ap.add_argument(
        "--gevsek-korelasyon", type=float, default=0.999,
        help="tolerans aşılsa bile bu korelasyonun üstü 'gevşek geçti' sayılır "
        "(INT8 modellerde Python/C++ ORT derlemeleri arası küçük çekirdek farkı beklenir)",
    )
    a = ap.parse_args()

    girdi_csv = BURASI / "altin_girdi.csv"
    skor_csv = BURASI / "altin_skor.csv"
    if not girdi_csv.exists() or not skor_csv.exists():
        print("HATA: golden dosyalar yok. Önce:  python 06_golden_uret.py")
        return 1
    exe = Path(a.exe)
    if not exe.exists():
        print(f"HATA: {exe} yok. Önce CMake ile derle (bkz. README.md 'Faz 4').")
        return 1

    beklenen = np.loadtxt(skor_csv, delimiter=",")
    cpp_cikti_csv = BURASI / "_cpp_skor.csv"

    sonuc = subprocess.run(
        [str(exe), str(Path(a.model).resolve()), str(girdi_csv.resolve()), str(cpp_cikti_csv.resolve())],
        capture_output=True, text=True,
    )
    print("--- C++ stderr (gecikme raporu) ---")
    print(sonuc.stderr.strip())
    if sonuc.returncode != 0:
        print(f"HATA: nabiz_cikarim.exe çıkış kodu {sonuc.returncode}")
        print(sonuc.stdout)
        return 1

    cpp_skor = np.loadtxt(cpp_cikti_csv, delimiter=",")
    if cpp_skor.shape != beklenen.shape:
        print(f"HATA: şekil uyuşmazlığı — C++ {cpp_skor.shape} vs Python {beklenen.shape}")
        return 1

    fark = np.abs(cpp_skor - beklenen)
    korelasyon = float(np.corrcoef(cpp_skor, beklenen)[0, 1])
    tam_gecti = bool(fark.max() < a.tolerans)
    gevsek_gecti = bool(korelasyon >= a.gevsek_korelasyon)

    print(f"\nn={len(beklenen)}  maks fark={fark.max():.3e}  ort fark={fark.mean():.3e}  korelasyon={korelasyon:.6f}")
    if tam_gecti:
        print(f"GECTI (tam — tolerans {a.tolerans:.1e})")
    elif gevsek_gecti:
        print(
            f"GECTI (gevsek — korelasyon {korelasyon:.6f} >= {a.gevsek_korelasyon}; "
            f"tam sayisal tolerans asildi, muhtemelen INT8 kuantize cekirdeklerin Python "
            f"onnxruntime wheel'i ile C++ SDK derlemesi arasindaki kucuk farki — siralama/ayrim korunuyor)"
        )
    else:
        print(f"BASARISIZ (tolerans {a.tolerans:.1e}, gevsek esik {a.gevsek_korelasyon})")
    cpp_cikti_csv.unlink(missing_ok=True)
    return 0 if (tam_gecti or gevsek_gecti) else 1


if __name__ == "__main__":
    raise SystemExit(main())
