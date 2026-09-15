"""Test katmanı 5 — yerel tampon (WAL), MQTT'ye bağımlılık yok (RB-03)."""
from nabiz.protocol.tampon import YerelTampon


def test_ekle_ve_boyut():
    t = YerelTampon()
    assert t.boyut() == 0
    t.ekle("m1", b"veri1")
    t.ekle("m2", b"veri2")
    assert t.boyut() == 2


def test_hepsi_sira_ve_icerik():
    t = YerelTampon()
    id1 = t.ekle("m1", b"a")
    id2 = t.ekle("m2", b"b")
    satirlar = t.hepsi()
    assert [s[0] for s in satirlar] == [id1, id2]
    assert satirlar[0][1:] == ("m1", b"a")
    assert satirlar[1][1:] == ("m2", b"b")


def test_sil():
    t = YerelTampon()
    id1 = t.ekle("m1", b"a")
    t.ekle("m2", b"b")
    t.sil(id1)
    assert t.boyut() == 1
    assert t.hepsi()[0][1] == "m2"


def test_temizle():
    t = YerelTampon()
    t.ekle("m1", b"a")
    t.ekle("m2", b"b")
    t.temizle()
    assert t.boyut() == 0


def test_dosya_tabanli_kalicilik(tmp_path):
    yol = tmp_path / "tampon.db"
    t1 = YerelTampon(yol)
    t1.ekle("m1", b"kalici-veri")
    t1.kapat()

    t2 = YerelTampon(yol)
    assert t2.boyut() == 1
    assert t2.hepsi()[0][1:] == ("m1", b"kalici-veri")


def test_bagimsiz_ornekler_izole():
    a, b = YerelTampon(), YerelTampon()
    a.ekle("m1", b"a")
    assert a.boyut() == 1
    assert b.boyut() == 0
