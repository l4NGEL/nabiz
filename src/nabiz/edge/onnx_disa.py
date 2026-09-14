"""ONNX export + PyTorch↔ONNX Runtime parite doğrulama.

Faz 2 torch modellerini ONNX'e aktarır; sayısal ve anomali-skoru paritesini
doğrular. RB-04 (nicemleme paritesi) ve Faz 4 (C++ ONNX Runtime çıkarımı)
buradan beslenir. Dinamik yığın (batch) ekseni — tek örnek de N örnek de çalışır.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def disa_aktar(
    agi: torch.nn.Module,
    ornek_girdi: np.ndarray | tuple[np.ndarray, ...],
    yol: str | Path,
    girdi_adlari: list[str] | None = None,
    opset: int = 17,
) -> Path:
    """Eğitilmiş bir ağı ONNX'e aktar. `ornek_girdi` tek dizi veya (çok-girdili
    modeller için, ör. FusionAE) dizi demeti olabilir."""
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    agi = agi.to("cpu").eval()

    coklu = isinstance(ornek_girdi, tuple)
    diziler = ornek_girdi if coklu else (ornek_girdi,)
    girdi_adlari = girdi_adlari or [f"girdi_{i}" for i in range(len(diziler))]
    tensorler = tuple(torch.tensor(np.asarray(x, dtype=np.float32)) for x in diziler)

    dinamik = {ad: {0: "yigin"} for ad in girdi_adlari}
    dinamik["cikti"] = {0: "yigin"}

    # dynamo=False: kasıtlı — TorchScript tabanlı legacy exporter (torch 2.9+'ta
    # varsayılan artık dynamo-tabanlı, kullanımdan kaldırılma uyarısı verir).
    # ConvAE/FusionAE'deki koşullu kırpma (Python bool -> Tensor karşılaştırma)
    # dynamo exporter'da farklı davranabilir; legacy yol burada doğrulanmış ve
    # kararlı. Geçiş: docs/yol-haritasi.md açık madde.
    with torch.no_grad():
        torch.onnx.export(
            agi,
            tensorler,
            str(yol),
            input_names=girdi_adlari,
            output_names=["cikti"],
            dynamic_axes=dinamik,
            opset_version=opset,
            dynamo=False,
        )
    return yol


def onnx_oturumu(yol: str | Path, saglayicilar: list[str] | None = None) -> ort.InferenceSession:
    saglayicilar = saglayicilar or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(yol), providers=saglayicilar)


def onnx_calistir(oturum: ort.InferenceSession, *girdiler: np.ndarray) -> np.ndarray:
    """Oturumun beklediği girdi adlarına göre besle, tek çıktıyı döndür."""
    adlar = [g.name for g in oturum.get_inputs()]
    besleme = {ad: np.asarray(x, dtype=np.float32) for ad, x in zip(adlar, girdiler, strict=True)}
    return oturum.run(None, besleme)[0]


def pariteyi_dogrula(
    agi: torch.nn.Module,
    oturum: ort.InferenceSession,
    *girdiler: np.ndarray,
    tolerans: float = 1e-4,
) -> dict:
    """PyTorch çıktısı vs ONNX Runtime çıktısı — maks/ort mutlak fark + korelasyon."""
    agi = agi.to("cpu").eval()
    with torch.no_grad():
        y_torch = agi(*(torch.tensor(np.asarray(x, dtype=np.float32)) for x in girdiler))
        if isinstance(y_torch, tuple):
            y_torch = y_torch[0]
        y_torch = y_torch.numpy()
    y_onnx = onnx_calistir(oturum, *girdiler)
    fark = np.abs(y_torch - y_onnx)
    korelasyon = float(np.corrcoef(y_torch.ravel(), y_onnx.ravel())[0, 1])
    return {
        "maks_mutlak_fark": float(fark.max()),
        "ort_mutlak_fark": float(fark.mean()),
        "korelasyon": korelasyon,
        "gecti": bool(fark.max() < tolerans),
    }


def yeniden_yapilandirma_skoru(y_gercek: np.ndarray, y_tahmin: np.ndarray) -> np.ndarray:
    """Örnek başına MSE — yığın dışı tüm eksenler üzerinden (AE anomali skoru)."""
    eksen = tuple(range(1, y_gercek.ndim))
    return ((y_tahmin - y_gercek) ** 2).mean(axis=eksen)


def skor_paritesi(y_gercek: np.ndarray, y_torch: np.ndarray, y_onnx: np.ndarray) -> dict:
    """İki çıktıdan türetilen anomali skorlarının (yeniden yapılandırma MSE) tutarlılığı."""
    s_t = yeniden_yapilandirma_skoru(y_gercek, y_torch)
    s_o = yeniden_yapilandirma_skoru(y_gercek, y_onnx)
    korelasyon = float(np.corrcoef(s_t, s_o)[0, 1]) if len(s_t) > 1 else 1.0
    return {"skor_korelasyon": korelasyon, "maks_skor_farki": float(np.abs(s_t - s_o).max())}


def dosya_boyutu_mb(yol: str | Path) -> float:
    return Path(yol).stat().st_size / (1024 * 1024)
