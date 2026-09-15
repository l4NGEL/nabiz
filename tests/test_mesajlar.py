"""Test katmanı 5 — tespit mesajı (Protobuf) oluşturma/serileştirme (IR-01)."""
import pytest

pytest.importorskip("google.protobuf")

from nabiz.protocol import mesajlar as M  # noqa: E402


def test_tespit_olustur_varsayilanlar():
    t = M.tespit_olustur("uc-01", "fan/id_00", 3.14)
    assert t.dugum_id == "uc-01"
    assert t.makine_id == "fan/id_00"
    assert t.anomali_skoru == pytest.approx(3.14, abs=1e-4)
    assert t.sinif == "anomali"
    assert t.siddet == "yok"
    assert len(t.mesaj_id) == 36  # uuid4 metin biçimi
    assert t.zaman_damgasi_ms > 0


def test_tespit_olustur_oznitelik_anlik_goruntu():
    t = M.tespit_olustur("uc-01", "fan/id_00", 1.0, oznitelik=[0.1, 0.2, 0.3])
    assert list(t.oznitelik_anlik_goruntu) == pytest.approx([0.1, 0.2, 0.3])


def test_tespit_olustur_gecersiz_siddet_hata():
    with pytest.raises(ValueError, match="siddet"):
        M.tespit_olustur("uc-01", "fan", 1.0, siddet="yokboyle")


def test_tespit_olustur_gecersiz_guven_hata():
    with pytest.raises(ValueError, match="guven"):
        M.tespit_olustur("uc-01", "fan", 1.0, guven=1.5)


def test_serilestir_coz_roundtrip():
    t = M.tespit_olustur(
        "uc-02", "pump/id_01", 5.5, sinif="anomali", guven=0.8, siddet="orta",
        oznitelik=[1.0, 2.0], model_surumu="dense_ae-v1", kaynak_split="birim_disi",
        zaman_damgasi_ms=1234567890, mesaj_id="sabit-id",
    )
    veri = M.serilestir(t)
    assert isinstance(veri, bytes) and len(veri) > 0
    t2 = M.coz(veri)
    assert t2.mesaj_id == "sabit-id"
    assert t2.dugum_id == "uc-02"
    assert t2.makine_id == "pump/id_01"
    assert t2.anomali_skoru == pytest.approx(5.5, abs=1e-4)
    assert t2.guven == pytest.approx(0.8, abs=1e-4)
    assert t2.siddet == "orta"
    assert list(t2.oznitelik_anlik_goruntu) == pytest.approx([1.0, 2.0])
    assert t2.model_surumu == "dense_ae-v1"
    assert t2.kaynak_split == "birim_disi"
    assert t2.zaman_damgasi_ms == 1234567890


def test_sozluge_cevir_ve_geri():
    t = M.tespit_olustur("uc-01", "fan/id_00", 2.0, mesaj_id="x", oznitelik=[1.0])
    d = M.sozluge_cevir(t)
    assert d["mesaj_id"] == "x"
    assert d["makine_id"] == "fan/id_00"
    t2 = M.sozlukten_olustur(d)
    assert t2.mesaj_id == "x"
    assert t2.anomali_skoru == pytest.approx(2.0, abs=1e-4)


def test_farkli_cagrilar_farkli_mesaj_id():
    a = M.tespit_olustur("uc-01", "fan", 1.0)
    b = M.tespit_olustur("uc-01", "fan", 1.0)
    assert a.mesaj_id != b.mesaj_id
