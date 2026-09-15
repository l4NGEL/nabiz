"""Test katmanı 5 — merkezi toplayıcı (FastAPI + SQLite, FR-08/IR-03)."""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from nabiz.serving.toplayici import uygulama_olustur  # noqa: E402


@pytest.fixture
def istemci():
    app = uygulama_olustur(":memory:")
    return TestClient(app)


def _govde(**over):
    taban = {
        "mesaj_id": "m1",
        "zaman_damgasi_ms": 1_700_000_000_000,
        "dugum_id": "uc-01",
        "makine_id": "fan/id_00",
        "anomali_skoru": 3.5,
        "sinif": "anomali",
        "guven": 0.8,
        "siddet": "orta",
        "oznitelik_anlik_goruntu": [0.1, 0.2],
        "model_surumu": "dense_ae-v1",
        "kaynak_split": "birim_disi",
    }
    taban.update(over)
    return taban


def test_tespit_kaydet_ve_getir(istemci):
    r = istemci.post("/tespit", json=_govde())
    assert r.status_code == 200
    assert r.json() == {"durum": "kaydedildi", "mesaj_id": "m1"}

    r2 = istemci.get("/tespit/m1")
    assert r2.status_code == 200
    satir = r2.json()
    assert satir["makine_id"] == "fan/id_00"
    assert satir["anomali_skoru"] == pytest.approx(3.5)


def test_ayni_mesaj_id_idempotent(istemci):
    istemci.post("/tespit", json=_govde())
    r2 = istemci.post("/tespit", json=_govde())
    assert r2.json()["durum"] == "yinelenen"
    tumu = istemci.get("/tespit").json()
    assert len(tumu) == 1  # ikinci kez yazılmadı


def test_bilinmeyen_mesaj_id_404(istemci):
    r = istemci.get("/tespit/yok-boyle")
    assert r.status_code == 404


def test_sorgula_makine_filtresi(istemci):
    istemci.post("/tespit", json=_govde(mesaj_id="m1", makine_id="fan/id_00"))
    istemci.post("/tespit", json=_govde(mesaj_id="m2", makine_id="pump/id_00"))
    sonuc = istemci.get("/tespit", params={"makine_id": "fan/id_00"}).json()
    assert len(sonuc) == 1
    assert sonuc[0]["makine_id"] == "fan/id_00"


def test_sorgula_zaman_araligi(istemci):
    istemci.post("/tespit", json=_govde(mesaj_id="eski", zaman_damgasi_ms=1000))
    istemci.post("/tespit", json=_govde(mesaj_id="yeni", zaman_damgasi_ms=5000))
    sonuc = istemci.get("/tespit", params={"baslangic_ms": 2000}).json()
    assert [s["mesaj_id"] for s in sonuc] == ["yeni"]


def test_sorgula_en_yeni_once(istemci):
    istemci.post("/tespit", json=_govde(mesaj_id="a", zaman_damgasi_ms=1000))
    istemci.post("/tespit", json=_govde(mesaj_id="b", zaman_damgasi_ms=2000))
    sonuc = istemci.get("/tespit").json()
    assert [s["mesaj_id"] for s in sonuc] == ["b", "a"]


def test_istatistik_makine_bazinda(istemci):
    istemci.post("/tespit", json=_govde(mesaj_id="a", makine_id="fan/id_00", anomali_skoru=1.0))
    istemci.post("/tespit", json=_govde(mesaj_id="b", makine_id="fan/id_00", anomali_skoru=3.0))
    istemci.post("/tespit", json=_govde(mesaj_id="c", makine_id="pump/id_00", anomali_skoru=9.0))
    ist = {s["makine_id"]: s for s in istemci.get("/istatistik").json()}
    assert ist["fan/id_00"]["adet"] == 2
    assert ist["fan/id_00"]["maks_skor"] == pytest.approx(3.0)
    assert ist["pump/id_00"]["adet"] == 1


def test_saglik(istemci):
    istemci.post("/tespit", json=_govde())
    r = istemci.get("/saglik")
    assert r.json() == {"durum": "ayakta", "toplam_kayit": 1}


def test_eksik_zorunlu_alan_422(istemci):
    govde = _govde()
    del govde["mesaj_id"]
    r = istemci.post("/tespit", json=govde)
    assert r.status_code == 422
