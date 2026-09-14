"""Gecikme / verim ölçümü — ONNX Runtime oturumu için (PR-02, PR-05).

Tek-örnek (yığın=1) çıkarım gecikmesi ölçülür — uç düğümde gerçek zamanlı
tespit senaryosu budur. Isınma turları hariç tutulur.
"""
from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort


def gecikme_olc(
    oturum: ort.InferenceSession,
    X: np.ndarray,
    n: int = 500,
    isinma: int = 20,
    seed: int = 0,
) -> dict:
    """Tek-örnek çıkarım gecikmesi. Döner: p50/p95/p99 (ms) + verim (örnek/sn)."""
    if len(X) < 1:
        raise ValueError("X boş")
    ad = oturum.get_inputs()[0].name
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), n + isinma)

    for i in idx[:isinma]:
        oturum.run(None, {ad: X[i : i + 1]})

    sureler_ms = np.empty(n)
    for k, i in enumerate(idx[isinma:]):
        t0 = time.perf_counter()
        oturum.run(None, {ad: X[i : i + 1]})
        sureler_ms[k] = (time.perf_counter() - t0) * 1000.0

    return {
        "n": n,
        "p50_ms": float(np.percentile(sureler_ms, 50)),
        "p95_ms": float(np.percentile(sureler_ms, 95)),
        "p99_ms": float(np.percentile(sureler_ms, 99)),
        "ort_ms": float(sureler_ms.mean()),
        "verim_ornek_sn": float(1000.0 / sureler_ms.mean()),
    }
