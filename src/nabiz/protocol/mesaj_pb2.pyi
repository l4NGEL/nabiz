from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Tespit(_message.Message):
    __slots__ = ("mesaj_id", "zaman_damgasi_ms", "dugum_id", "makine_id", "anomali_skoru", "sinif", "guven", "siddet", "oznitelik_anlik_goruntu", "model_surumu", "kaynak_split")
    MESAJ_ID_FIELD_NUMBER: _ClassVar[int]
    ZAMAN_DAMGASI_MS_FIELD_NUMBER: _ClassVar[int]
    DUGUM_ID_FIELD_NUMBER: _ClassVar[int]
    MAKINE_ID_FIELD_NUMBER: _ClassVar[int]
    ANOMALI_SKORU_FIELD_NUMBER: _ClassVar[int]
    SINIF_FIELD_NUMBER: _ClassVar[int]
    GUVEN_FIELD_NUMBER: _ClassVar[int]
    SIDDET_FIELD_NUMBER: _ClassVar[int]
    OZNITELIK_ANLIK_GORUNTU_FIELD_NUMBER: _ClassVar[int]
    MODEL_SURUMU_FIELD_NUMBER: _ClassVar[int]
    KAYNAK_SPLIT_FIELD_NUMBER: _ClassVar[int]
    mesaj_id: str
    zaman_damgasi_ms: int
    dugum_id: str
    makine_id: str
    anomali_skoru: float
    sinif: str
    guven: float
    siddet: str
    oznitelik_anlik_goruntu: _containers.RepeatedScalarFieldContainer[float]
    model_surumu: str
    kaynak_split: str
    def __init__(self, mesaj_id: _Optional[str] = ..., zaman_damgasi_ms: _Optional[int] = ..., dugum_id: _Optional[str] = ..., makine_id: _Optional[str] = ..., anomali_skoru: _Optional[float] = ..., sinif: _Optional[str] = ..., guven: _Optional[float] = ..., siddet: _Optional[str] = ..., oznitelik_anlik_goruntu: _Optional[_Iterable[float]] = ..., model_surumu: _Optional[str] = ..., kaynak_split: _Optional[str] = ...) -> None: ...
