"""Motor kalan ömür (RUL) tahmini + bakım risk sıralaması API'si (FastAPI).

Öznitelikler her istekte DB'den, eğitimdekiyle AYNI SQL pencere sorgusuyla
(`data/sql_oznitelik.py`) motor başına son çevrim için üretilir — eğitim/sunum
arasında öznitelik kayması (training-serving skew) olmaz.

Uç noktalar:
  GET /saglik
  GET /motorlar?kaynak=FD001-test&limit=20   en riskli önce (tahmini RUL artan)
  GET /motorlar/{birim}?kaynak=FD001-test    tek motor

Çalıştırma:
  set NABIZ_RUL_DSN=postgresql://nabiz:nabiz_yerel@localhost:55432/nabiz
  set NABIZ_RUL_MODEL=modeller/rul/hgb_fd001.joblib
  uvicorn nabiz.serving.rul_api:ortamdan_uygulama --factory --app-dir src --port 8710
"""
from __future__ import annotations

import os
import threading
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from nabiz import config
from nabiz.data import sql_oznitelik as S
from nabiz.models import rul_risk as R


class MotorTahmini(BaseModel):
    sira: int | None = None
    birim: int
    son_cevrim: int
    tahmini_rul: float
    risk: str


def uygulama_olustur(dsn: str, model_yolu: str | os.PathLike, artefakt: dict[str, Any] | None = None) -> FastAPI:
    """`artefakt` verilirse (testler) diskten model okunmaz."""
    art = artefakt if artefakt is not None else joblib.load(model_yolu)
    model, sensorler, oznitelikler = art["model"], art["sensorler"], art["oznitelikler"]
    baglanti = S.baglan(dsn)
    kilit = threading.Lock()  # tek bağlantı; FastAPI sync uç noktaları iş parçacığında koşar

    app = FastAPI(title="NABIZ RUL / bakım önceliklendirme", version="1.0")

    def _tahminler(kaynak: str, birim: int | None = None) -> list[MotorTahmini]:
        with kilit:
            oz = S.oznitelik_cek(baglanti, sensorler, kaynak, birim=birim, yalniz_son_cevrim=True)
        if oz.empty:
            return []
        rul = np.clip(model.predict(oz[oznitelikler].to_numpy()), 0, None)
        return [
            MotorTahmini(birim=int(b), son_cevrim=int(c), tahmini_rul=round(float(r), 1), risk=R.risk_seviyesi(r))
            for b, c, r in zip(oz["birim"], oz["cevrim"], rul, strict=True)
        ]

    @app.get("/saglik")
    def saglik() -> dict[str, Any]:
        return {"durum": "ok", "model": art.get("yapilandirma"), "hedef": art.get("hedef"),
                "sensor_sayisi": len(sensorler), "db": baglanti.lehce}

    @app.get("/motorlar", response_model=list[MotorTahmini])
    def motorlar(kaynak: str = Query(...), limit: int = Query(20, ge=1, le=1000)) -> list[MotorTahmini]:
        tahminler = _tahminler(kaynak)
        if not tahminler:
            raise HTTPException(404, f"kaynak bulunamadı ya da boş: {kaynak}")
        sirali = R.risk_sirala([t.birim for t in tahminler], [t.tahmini_rul for t in tahminler])
        cevrim = {t.birim: t.son_cevrim for t in tahminler}
        return [
            MotorTahmini(sira=int(r.sira), birim=int(r.birim), son_cevrim=cevrim[int(r.birim)],
                         tahmini_rul=float(r.tahmini_rul), risk=str(r.risk))
            for r in sirali.head(limit).itertuples()
        ]

    @app.get("/motorlar/{birim}", response_model=MotorTahmini)
    def motor(birim: int, kaynak: str = Query(...)) -> MotorTahmini:
        tahminler = _tahminler(kaynak, birim)
        if not tahminler:
            raise HTTPException(404, f"motor bulunamadı: {kaynak}/{birim}")
        return tahminler[0]

    app.state.baglanti = baglanti
    return app


def ortamdan_uygulama() -> FastAPI:
    """uvicorn --factory girişi: NABIZ_RUL_DSN + NABIZ_RUL_MODEL ortam değişkenleri."""
    dsn = os.environ.get("NABIZ_RUL_DSN") or os.environ.get("NABIZ_PG_DSN")
    if not dsn:
        raise RuntimeError("NABIZ_RUL_DSN (ya da NABIZ_PG_DSN) tanımlı değil")
    model = os.environ.get("NABIZ_RUL_MODEL", str(config.MODEL_DIZINI / "rul" / "hgb_fd001.joblib"))
    return uygulama_olustur(dsn, model)
