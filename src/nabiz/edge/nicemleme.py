"""ONNX Runtime dinamik nicemleme (INT8) — RB-04.

PTQ (post-training quantization): ağırlıklar INT8'e nicemlenir, aktivasyonlar
çalışma zamanında dinamik ölçeklenir. Eğitim/etiketli kalibrasyon verisi
gerektirmez (QAT değil) — küçük modeller ve tek-örnek uç çıkarımı için yeterli.
Doğruluk paritesi (AUC düşüşü, skor korelasyonu) `03`/`04` betiklerinde ölçülür.
"""
from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def int8_nicemle(
    kaynak_onnx: str | Path,
    hedef_onnx: str | Path,
    op_turleri: list[str] | None = ("MatMul", "Gemm"),
) -> Path:
    """FP32 ONNX modelini dinamik INT8'e nicemle (ağırlıklar).

    `op_turleri` öntanımı yalnız MatMul/Gemm (Linear katmanları) — dinamik INT8
    nicemleme Conv2d için ONNX Runtime CPU EP'de her zaman desteklenmiyor
    (`ConvInteger` NOT_IMPLEMENTED). Conv ağırlıklı modeller (ör. ConvAE) için
    bu, yalnız darboğaz/Linear katmanları nicemler — çalışır ama daha az
    sıkıştırır; tam Conv nicemleme statik (kalibrasyon verili) nicemleme
    ister, Faz 3 açık maddesi. `op_turleri=None` verirse kısıt kalkar.
    """
    kaynak, hedef = Path(kaynak_onnx), Path(hedef_onnx)
    if not kaynak.exists():
        raise FileNotFoundError(kaynak)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    kw = {"op_types_to_quantize": list(op_turleri)} if op_turleri else {}
    quantize_dynamic(model_input=str(kaynak), model_output=str(hedef), weight_type=QuantType.QInt8, **kw)
    return hedef


def dosya_boyutu_mb(yol: str | Path) -> float:
    yol = Path(yol)
    if not yol.exists():
        raise FileNotFoundError(yol)
    return yol.stat().st_size / (1024 * 1024)
