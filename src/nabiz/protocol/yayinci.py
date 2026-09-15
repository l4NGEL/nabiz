"""MQTT tespit yayıncısı — yerel tampon + backfill (RB-03, IR-02).

Broker ulaşılamazsa (başlangıçta kapalı, ya da akış ortasında düşerse) mesajlar
`tampon.YerelTampon`'a yazılır; `on_connect` (ilk bağlantı VEYA yeniden bağlanma)
tetiklendiğinde kuyruk sırayla gönderilip boşaltılır. QoS 1 + `mesaj_id`
idempotency ile çift teslim merkezi toplayıcıda güvenli (bkz. serving/toplayici.py
— `mesaj_id` PRIMARY KEY, INSERT OR IGNORE).

`istemci` enjekte edilebilir (test double) — broker olmadan tam birim testi.
"""
from __future__ import annotations

import threading
from typing import Any, Protocol

from nabiz.protocol import mesajlar as M
from nabiz.protocol.tampon import YerelTampon


class _MQTTIstemcisi(Protocol):
    on_connect: Any
    on_disconnect: Any

    def connect_async(self, host: str, port: int) -> None: ...
    def loop_start(self) -> None: ...
    def loop_stop(self) -> None: ...
    def disconnect(self) -> None: ...
    def publish(self, konu: str, veri: bytes, qos: int) -> Any: ...


def _gercek_istemci_olustur(client_id: str | None):
    import paho.mqtt.client as mqtt

    return mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id or "",
    )


class TespitYayincisi:
    ad = "mqtt_yayinci"

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        konu: str = "nabiz/tespit",
        tampon_yolu: str = ":memory:",
        qos: int = 1,
        client_id: str | None = None,
        istemci: _MQTTIstemcisi | None = None,
    ) -> None:
        self.konu = konu
        self.qos = qos
        self._broker_host, self._broker_port = broker_host, broker_port
        self.tampon = YerelTampon(tampon_yolu)
        self._baglandi = False
        self._kilit = threading.Lock()
        self._backfill_sayaci = 0

        self._istemci: _MQTTIstemcisi = istemci or _gercek_istemci_olustur(client_id)
        self._istemci.on_connect = self._baglanildi
        self._istemci.on_disconnect = self._baglanti_koptu

    # --- bağlantı yaşam döngüsü -------------------------------------------------
    def baglan(self) -> None:
        self._istemci.connect_async(self._broker_host, self._broker_port)
        self._istemci.loop_start()

    def kapat(self) -> None:
        self._istemci.loop_stop()
        try:
            self._istemci.disconnect()
        except Exception:  # noqa: BLE001 — kapatma sırasında bağlantı zaten kopuk olabilir
            pass
        self.tampon.kapat()

    def _baglanildi(self, *args, **kwargs) -> None:  # paho v2 imzası: (client, userdata, flags, reason_code, properties)
        with self._kilit:
            self._baglandi = True
        self._backfill_gonder()

    def _baglanti_koptu(self, *args, **kwargs) -> None:
        with self._kilit:
            self._baglandi = False

    @property
    def baglandi_mi(self) -> bool:
        with self._kilit:
            return self._baglandi

    # --- yayınlama ---------------------------------------------------------------
    def yayinla(self, tespit) -> bool:
        """True: broker'a doğrudan gönderildi. False: tampona yazıldı (broker yok/hata)."""
        veri = M.serilestir(tespit)
        if self.baglandi_mi:
            sonuc = self._istemci.publish(self.konu, veri, qos=self.qos)
            if getattr(sonuc, "rc", 1) == 0:  # mqtt.MQTT_ERR_SUCCESS == 0
                return True
        self.tampon.ekle(tespit.mesaj_id, veri)
        return False

    def _backfill_gonder(self) -> None:
        for id_, _mesaj_id, veri in self.tampon.hepsi():
            sonuc = self._istemci.publish(self.konu, veri, qos=self.qos)
            if getattr(sonuc, "rc", 1) == 0:
                self.tampon.sil(id_)
                self._backfill_sayaci += 1
            else:
                break  # sıralı gönderim — biri başarısızsa dur, sıradaki denemede devam

    @property
    def backfill_sayaci(self) -> int:
        return self._backfill_sayaci
