"""Test katmanı 5 — RUL risk sıralaması (rul_risk) + API (FastAPI + SQLite)."""
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from nabiz.data import cmapss as C
from nabiz.data import sql_oznitelik as S
from nabiz.models import rul_risk as R

SENSORLER = ["sensor_2", "sensor_7"]


# ------------------------------------------------------------------ rul_risk


@pytest.mark.parametrize(
    ("rul", "seviye"), [(0, "KRITIK"), (30, "KRITIK"), (30.1, "YUKSEK"), (60, "YUKSEK"), (90, "IZLE"), (91, "NORMAL")]
)
def test_risk_seviyesi_esikleri(rul, seviye):
    assert R.risk_seviyesi(rul) == seviye


def test_risk_seviyesi_nan_reddeder():
    with pytest.raises(ValueError):
        R.risk_seviyesi(float("nan"))


def test_risk_sirala_en_riskli_once():
    df = R.risk_sirala([7, 3, 9], [80.0, 12.0, 45.0])
    assert list(df["birim"]) == [3, 9, 7]
    assert list(df["sira"]) == [1, 2, 3]
    assert list(df["risk"]) == ["KRITIK", "YUKSEK", "IZLE"]


def test_siralama_metrikleri_mukemmel_ve_ters():
    g = np.array([5.0, 20.0, 50.0, 100.0, 120.0])
    iyi = R.siralama_metrikleri(g, g, ust_k=2)
    assert iyi["spearman"] == pytest.approx(1.0)
    assert iyi["kritik_yakalama"] == 1.0 and iyi["kritik_kesinlik_ust_k"] == 1.0
    ters = R.siralama_metrikleri(g, g[::-1], ust_k=2)
    assert ters["spearman"] == pytest.approx(-1.0)
    assert ters["kritik_yakalama"] == 0.0  # kritik motorlar 120/100 tahmin edildi -> kaçırıldı


def test_seviye_karisiklik_boyut():
    tablo = R.seviye_karisiklik([10, 70, 200], [15, 100, 200])
    assert tablo.shape == (4, 4)
    assert int(tablo.to_numpy().sum()) == 3
    assert tablo.loc["IZLE", "NORMAL"] == 1


# ------------------------------------------------------------------ API

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nabiz.serving.rul_api import uygulama_olustur  # noqa: E402


@pytest.fixture
def istemci(tmp_path):
    veri = C.sentetik_uret(n_birim=8, min_cevrim=40, maks_cevrim=80, seed=2)
    db = tmp_path / "rul.db"
    b = S.baglan(f"sqlite:///{db}")
    S.sema_olustur(b)
    S.yukle(b, veri, "egitim")
    S.yukle(b, veri[veri["cevrim"] <= 35], "kesik")  # "test" motorları: 35. çevrimde kesik
    oz = S.oznitelik_cek(b, SENSORLER, "egitim")
    b.kapat()
    y = np.minimum(oz.groupby("birim")["cevrim"].transform("max") - oz["cevrim"], 125).to_numpy(float)
    adlar = S.oznitelik_adlari(SENSORLER)
    model = HistGradientBoostingRegressor(max_iter=50, random_state=0).fit(oz[adlar].to_numpy(), y)
    art = {"model": model, "sensorler": SENSORLER, "oznitelikler": adlar, "yapilandirma": "test", "hedef": "sentetik"}
    return TestClient(uygulama_olustur(f"sqlite:///{db}", "", artefakt=art))


def test_saglik(istemci):
    r = istemci.get("/saglik")
    assert r.status_code == 200
    assert r.json()["db"] == "sqlite" and r.json()["sensor_sayisi"] == 2


def test_motorlar_riske_gore_sirali(istemci):
    r = istemci.get("/motorlar", params={"kaynak": "kesik", "limit": 5})
    assert r.status_code == 200
    govde = r.json()
    assert len(govde) == 5
    rul = [m["tahmini_rul"] for m in govde]
    assert rul == sorted(rul)
    assert [m["sira"] for m in govde] == [1, 2, 3, 4, 5]
    assert all(m["son_cevrim"] == 35 for m in govde)
    assert all(m["risk"] == R.risk_seviyesi(m["tahmini_rul"]) for m in govde)


def test_tek_motor_ve_liste_tutarli(istemci):
    liste = {m["birim"]: m for m in istemci.get("/motorlar", params={"kaynak": "kesik", "limit": 100}).json()}
    tek = istemci.get("/motorlar/4", params={"kaynak": "kesik"}).json()
    assert tek["birim"] == 4
    assert tek["tahmini_rul"] == liste[4]["tahmini_rul"]


def test_bulunamayan_kaynak_ve_motor_404(istemci):
    assert istemci.get("/motorlar", params={"kaynak": "yok"}).status_code == 404
    assert istemci.get("/motorlar/999", params={"kaynak": "kesik"}).status_code == 404
