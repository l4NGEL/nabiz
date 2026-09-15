"""Test katmanı 5 — MQTT yayıncı + tampon/backfill (RB-03). Broker GEREKMEZ —
enjekte edilen sahte istemci (`_SahteIstemci`) ile tam mantık testi; gerçek
paho-mqtt/broker'a dokunmaz. Gerçek broker'lı uçtan uca doğrulama:
`docker compose up` + `07_protokol_demo.py` (bkz. README.md 'Faz 5')."""
from __future__ import annotations

import pytest

pytest.importorskip("google.protobuf")

from nabiz.protocol import mesajlar as M  # noqa: E402
from nabiz.protocol.yayinci import TespitYayincisi  # noqa: E402


class _SahteSonuc:
    def __init__(self, rc: int) -> None:
        self.rc = rc


class _SahteIstemci:
    """paho.mqtt.client.Client'ın yerine geçen minimal test double'ı."""

    def __init__(self, basarisiz_yayinla: bool = False) -> None:
        self.on_connect = None
        self.on_disconnect = None
        self.yayinlar: list[tuple[str, bytes, int]] = []
        self.baglan_cagrildi = False
        self.basarisiz_yayinla = basarisiz_yayinla

    def connect_async(self, host: str, port: int) -> None:
        self.baglan_cagrildi = True

    def loop_start(self) -> None:
        pass

    def loop_stop(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def publish(self, konu: str, veri: bytes, qos: int):
        self.yayinlar.append((konu, veri, qos))
        return _SahteSonuc(rc=1 if self.basarisiz_yayinla else 0)

    def baglandi_simule_et(self) -> None:
        self.on_connect(self, None, None, 0)

    def koptu_simule_et(self) -> None:
        self.on_disconnect(self, None, 0)


@pytest.fixture
def sahte():
    return _SahteIstemci()


def _yayinci(istemci) -> TespitYayincisi:
    return TespitYayincisi(istemci=istemci)


def test_baglanmadan_yayinla_tampona_yazar(sahte):
    y = _yayinci(sahte)
    t = M.tespit_olustur("uc-01", "fan", 1.0, mesaj_id="m1")
    sonuc = y.yayinla(t)
    assert sonuc is False
    assert y.tampon.boyut() == 1
    assert sahte.yayinlar == []  # broker'a hiç gitmedi


def test_baglandiktan_sonra_yayinla_dogrudan_gider(sahte):
    y = _yayinci(sahte)
    y.baglan()
    sahte.baglandi_simule_et()
    assert y.baglandi_mi is True

    t = M.tespit_olustur("uc-01", "fan", 1.0, mesaj_id="m1")
    sonuc = y.yayinla(t)
    assert sonuc is True
    assert y.tampon.boyut() == 0
    assert len(sahte.yayinlar) == 1
    assert sahte.yayinlar[0][0] == y.konu
    assert sahte.yayinlar[0][2] == 1  # qos


def test_baglanma_tetikler_backfill(sahte):
    y = _yayinci(sahte)
    # broker kapalıyken 3 mesaj tampona yazılsın
    for i in range(3):
        y.yayinla(M.tespit_olustur("uc-01", "fan", float(i), mesaj_id=f"m{i}"))
    assert y.tampon.boyut() == 3
    assert sahte.yayinlar == []

    y.baglan()
    sahte.baglandi_simule_et()  # on_connect -> backfill

    assert y.tampon.boyut() == 0
    assert len(sahte.yayinlar) == 3
    assert y.backfill_sayaci == 3


def test_baglanti_kopunca_tekrar_tamponlar(sahte):
    y = _yayinci(sahte)
    y.baglan()
    sahte.baglandi_simule_et()
    sahte.koptu_simule_et()
    assert y.baglandi_mi is False

    y.yayinla(M.tespit_olustur("uc-01", "fan", 1.0, mesaj_id="m1"))
    assert y.tampon.boyut() == 1


def test_yayin_basarisiz_olursa_tampona_duser():
    sahte = _SahteIstemci(basarisiz_yayinla=True)
    y = _yayinci(sahte)
    y.baglan()
    sahte.baglandi_simule_et()

    sonuc = y.yayinla(M.tespit_olustur("uc-01", "fan", 1.0, mesaj_id="m1"))
    assert sonuc is False
    assert y.tampon.boyut() == 1


def test_tamponlanan_mesaj_dogru_cozulur(sahte):
    y = _yayinci(sahte)
    orijinal = M.tespit_olustur("uc-01", "fan/id_00", 7.5, guven=0.9, mesaj_id="m1")
    y.yayinla(orijinal)

    _id, mesaj_id, veri = y.tampon.hepsi()[0]
    cozulen = M.coz(veri)
    assert mesaj_id == "m1"
    assert cozulen.makine_id == "fan/id_00"
    assert cozulen.anomali_skoru == pytest.approx(7.5, abs=1e-4)
    assert cozulen.guven == pytest.approx(0.9, abs=1e-4)


def test_kapat_calisir(sahte):
    y = _yayinci(sahte)
    y.baglan()
    y.kapat()  # istisna atmamalı
