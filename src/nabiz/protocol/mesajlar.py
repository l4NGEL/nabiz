"""Tespit mesajı oluşturma / (de)serileştirme — `mesaj.proto`nun Python sarmalayıcısı.

IR-01: alanlar şemada sabit numaralı; burada yalnız ergonomi (varsayılanlar,
zaman damgası otomatik, sözlük dönüşümü).
"""
from __future__ import annotations

import time
import uuid

from google.protobuf.json_format import MessageToDict, ParseDict

from nabiz.protocol.mesaj_pb2 import Tespit

_GECERLI_SINIFLAR = {"normal", "anomali"}
_GECERLI_SIDDETLER = {"yok", "baslangic", "orta", "ileri"}


def tespit_olustur(
    dugum_id: str,
    makine_id: str,
    anomali_skoru: float,
    sinif: str = "anomali",
    guven: float = 0.0,
    siddet: str = "yok",
    oznitelik: list[float] | None = None,
    model_surumu: str = "",
    kaynak_split: str = "",
    zaman_damgasi_ms: int | None = None,
    mesaj_id: str | None = None,
) -> Tespit:
    """Yeni bir `Tespit` protobuf mesajı kur (alan doğrulamasıyla)."""
    if siddet not in _GECERLI_SIDDETLER:
        raise ValueError(f"geçersiz siddet: {siddet!r} (beklenen: {sorted(_GECERLI_SIDDETLER)})")
    if not (0.0 <= guven <= 1.0):
        raise ValueError(f"guven [0,1] aralığında olmalı: {guven}")

    t = Tespit()
    t.mesaj_id = mesaj_id or str(uuid.uuid4())
    t.zaman_damgasi_ms = zaman_damgasi_ms if zaman_damgasi_ms is not None else int(time.time() * 1000)
    t.dugum_id = dugum_id
    t.makine_id = makine_id
    t.anomali_skoru = float(anomali_skoru)
    t.sinif = sinif
    t.guven = float(guven)
    t.siddet = siddet
    if oznitelik:
        t.oznitelik_anlik_goruntu.extend(float(x) for x in oznitelik)
    t.model_surumu = model_surumu
    t.kaynak_split = kaynak_split
    return t


def serilestir(tespit: Tespit) -> bytes:
    return tespit.SerializeToString()


def coz(veri: bytes) -> Tespit:
    t = Tespit()
    t.ParseFromString(veri)
    return t


def sozluge_cevir(tespit: Tespit) -> dict:
    """JSON/HTTP taşıma için (ör. serving/toplayici.py POST gövdesi)."""
    return MessageToDict(tespit, preserving_proto_field_name=True)


def sozlukten_olustur(d: dict) -> Tespit:
    t = Tespit()
    return ParseDict(d, t)
