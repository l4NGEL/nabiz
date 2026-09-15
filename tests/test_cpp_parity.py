"""Test katmanı 1/4 — C++ çıkarım ikilisi golden-vector paritesi.

CMake ile derlenmiş `nabiz_cikarim.exe` ve Faz 3'ün ürettiği ONNX modeli
YOKSA atlanır (bu makineye özgü — CI'da C++ toolchain/ONNX Runtime SDK yok).
Var olduğunda gerçek bir entegrasyon testi: 06_golden_uret.py + C++ ikilisi +
cpp/testler/dogrula.py'nin aynı akışını uçtan uca çalıştırır.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).parent.parent
EXE = KOK / "cpp" / "build" / "Release" / "nabiz_cikarim.exe"
MODEL = KOK / "modeller" / "edge" / "dense_ae_fp32.onnx"

pytestmark = pytest.mark.skipif(
    not (EXE.exists() and MODEL.exists()),
    reason="C++ ikilisi veya ONNX modeli yok — önce CMake build + 04_uc_uyarla.py",
)


def _calistir(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=KOK, capture_output=True, text=True, timeout=120
    )


def test_golden_uret_ve_dogrula_fp32():
    uret = _calistir("06_golden_uret.py", "--n", "20")
    assert uret.returncode == 0, uret.stdout + uret.stderr

    dogrula = _calistir("cpp/testler/dogrula.py")
    assert dogrula.returncode == 0, dogrula.stdout + dogrula.stderr
    assert "GECTI" in dogrula.stdout


def test_cpp_binary_dogrudan_calisir(tmp_path):
    girdi = KOK / "cpp" / "testler" / "altin_girdi.csv"
    if not girdi.exists():
        pytest.skip("altin_girdi.csv yok — önce 06_golden_uret.py")
    cikti = tmp_path / "skor.csv"
    sonuc = subprocess.run(
        [str(EXE), str(MODEL), str(girdi), str(cikti)],
        capture_output=True, text=True, timeout=30,
    )
    assert sonuc.returncode == 0, sonuc.stdout + sonuc.stderr
    assert cikti.exists()
    assert "P50=" in sonuc.stderr


def test_cpp_binary_eksik_argumanla_hata():
    sonuc = subprocess.run([str(EXE)], capture_output=True, text=True, timeout=10)
    assert sonuc.returncode != 0
    assert "Kullanim" in sonuc.stderr
