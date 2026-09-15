"""Merkezi tespit toplayıcı — FastAPI + SQLite (FR-08, IR-03).

Uç düğümlerden gelen tespitleri alır (HTTP POST /tespit — `protocol/mesajlar.py`
`sozluge_cevir` çıktısı), saklar, zaman-aralığı/makine sorgusu sunar.
`mesaj_id` PRIMARY KEY + INSERT OR IGNORE → idempotent (RB-03: QoS1 çift teslim
güvenli, aynı mesaj iki kez gelse de tek kayıt).

Gerçek dağıtımda bir MQTT subscriber (bkz. `07_protokol_demo.py`) bu endpoint'i
besler. Prodüksiyonda SQLite yerine TimescaleDB/Postgres — `db_yolu` ile
değiştirilebilir; şema/sorgu arayüzü aynı kalır.

Çalıştırma:  uvicorn nabiz.serving.toplayici:app --port 8700
"""
from __future__ import annotations

import json
import sqlite3
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nabiz import config


class TespitGovde(BaseModel):
    mesaj_id: str
    zaman_damgasi_ms: int
    dugum_id: str = ""
    makine_id: str = ""
    anomali_skoru: float = 0.0
    sinif: str = "anomali"
    guven: float = 0.0
    siddet: str = "yok"
    oznitelik_anlik_goruntu: list[float] = Field(default_factory=list)
    model_surumu: str = ""
    kaynak_split: str = ""


_SEMA = """
CREATE TABLE IF NOT EXISTS tespitler (
    mesaj_id TEXT PRIMARY KEY,
    zaman_damgasi_ms INTEGER NOT NULL,
    dugum_id TEXT, makine_id TEXT,
    anomali_skoru REAL, sinif TEXT, guven REAL, siddet TEXT,
    oznitelik_json TEXT, model_surumu TEXT, kaynak_split TEXT,
    alinma_zamani REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_zaman ON tespitler(zaman_damgasi_ms);
CREATE INDEX IF NOT EXISTS idx_makine ON tespitler(makine_id);
"""


def _db_kur(yol: str) -> sqlite3.Connection:
    db = sqlite3.connect(yol, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript(_SEMA)
    db.commit()
    return db


def uygulama_olustur(db_yolu: str = ":memory:") -> FastAPI:
    app = FastAPI(title="NABIZ Merkezi Toplayıcı")
    db = _db_kur(db_yolu)
    app.state.db = db

    @app.post("/tespit")
    def tespit_al(govde: TespitGovde) -> dict:
        onceki = db.execute(
            "SELECT 1 FROM tespitler WHERE mesaj_id = ?", (govde.mesaj_id,)
        ).fetchone()
        db.execute(
            """INSERT OR IGNORE INTO tespitler
               (mesaj_id, zaman_damgasi_ms, dugum_id, makine_id, anomali_skoru,
                sinif, guven, siddet, oznitelik_json, model_surumu, kaynak_split, alinma_zamani)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                govde.mesaj_id, govde.zaman_damgasi_ms, govde.dugum_id, govde.makine_id,
                govde.anomali_skoru, govde.sinif, govde.guven, govde.siddet,
                json.dumps(govde.oznitelik_anlik_goruntu), govde.model_surumu,
                govde.kaynak_split, time.time(),
            ),
        )
        db.commit()
        return {"durum": "yinelenen" if onceki else "kaydedildi", "mesaj_id": govde.mesaj_id}

    @app.get("/tespit/{mesaj_id}")
    def tekil_getir(mesaj_id: str) -> dict:
        satir = db.execute("SELECT * FROM tespitler WHERE mesaj_id = ?", (mesaj_id,)).fetchone()
        if satir is None:
            raise HTTPException(404, f"mesaj_id bulunamadı: {mesaj_id}")
        return dict(satir)

    @app.get("/tespit")
    def sorgula(
        makine_id: str | None = None,
        baslangic_ms: int | None = None,
        bitis_ms: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        kosullar, degerler = [], []
        if makine_id is not None:
            kosullar.append("makine_id = ?")
            degerler.append(makine_id)
        if baslangic_ms is not None:
            kosullar.append("zaman_damgasi_ms >= ?")
            degerler.append(baslangic_ms)
        if bitis_ms is not None:
            kosullar.append("zaman_damgasi_ms <= ?")
            degerler.append(bitis_ms)
        nerede = f"WHERE {' AND '.join(kosullar)}" if kosullar else ""
        sorgu = f"SELECT * FROM tespitler {nerede} ORDER BY zaman_damgasi_ms DESC LIMIT ?"
        satirlar = db.execute(sorgu, (*degerler, max(1, min(limit, 1000)))).fetchall()
        return [dict(s) for s in satirlar]

    @app.get("/istatistik")
    def istatistik() -> list[dict]:
        return [
            dict(s)
            for s in db.execute(
                """SELECT makine_id, COUNT(*) AS adet,
                          AVG(anomali_skoru) AS ort_skor, MAX(anomali_skoru) AS maks_skor,
                          MAX(zaman_damgasi_ms) AS son_zaman_ms
                   FROM tespitler GROUP BY makine_id ORDER BY maks_skor DESC"""
            ).fetchall()
        ]

    @app.get("/saglik")
    def saglik() -> dict:
        (toplam,) = db.execute("SELECT COUNT(*) FROM tespitler").fetchone()
        return {"durum": "ayakta", "toplam_kayit": toplam}

    return app


config.dizinleri_hazirla()
app = uygulama_olustur(str(config.VERI_KOK / "toplayici.db"))
