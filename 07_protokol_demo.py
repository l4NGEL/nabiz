"""Faz 5 — protokol demosu: tespit üret → yayınla (MQTT, broker yoksa tampona
yazılır) → varsa merkezi toplayıcıya HTTP ile de gönder → sonuçları raporla.

Broker (mosquitto) ve toplayıcı (uvicorn) GEREKMEZ — ikisi de yoksa demo yine
çalışır, yalnız "tamponda kaldı" / "toplayıcıya ulaşılamadı" der. Gerçek
uçtan uca için:

  # 1) MQTT broker (Docker)
  docker run -d -p 1883:1883 eclipse-mosquitto
  # 2) Merkezi toplayıcı
  uvicorn nabiz.serving.toplayici:app --app-dir src --port 8700
  # 3) Demo
  python 07_protokol_demo.py --broker-host localhost --toplayici-url http://localhost:8700

ya da tek komutla: docker compose up  (bkz. docker-compose.yml)

Çıktı:  sonuclar/faz5-protokol.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402

utf8_cikti()

try:
    from nabiz.protocol import mesajlar as M
    from nabiz.protocol.yayinci import TespitYayincisi

    HAZIR = True
except ImportError:
    HAZIR = False


def sentetik_tespitler_uret(n: int = 12, seed: int = 0) -> list:
    import numpy as np

    rng = np.random.default_rng(seed)
    tespitler = []
    for _i in range(n):
        anomali = bool(rng.random() < 0.3)
        skor = float(rng.uniform(3.0, 10.0) if anomali else rng.uniform(0.1, 1.0))
        tespitler.append(
            M.tespit_olustur(
                dugum_id="uc-demo-01",
                makine_id=str(rng.choice(["fan/id_00", "fan/id_01", "pump/id_00"])),
                anomali_skoru=skor,
                sinif="anomali" if anomali else "normal",
                guven=float(rng.uniform(0.6, 0.99)),
                siddet=str(rng.choice(["baslangic", "orta", "ileri"])) if anomali else "yok",
                oznitelik=list(rng.normal(0, 1, 8)),
                model_surumu="dense_ae-fp32-demo",
                kaynak_split="birim_disi",
            )
        )
    return tespitler


def toplayiciya_gonder(tespitler: list, url: str) -> dict:
    try:
        import httpx
    except ImportError:
        return {"ulasildi": False, "sebep": "httpx kurulu değil"}

    gonderilen, hatali = 0, 0
    try:
        with httpx.Client(timeout=2.0) as istemci:
            istemci.get(f"{url}/saglik")  # erişilebilirlik kontrolü
            for t in tespitler:
                yanit = istemci.post(f"{url}/tespit", json=M.sozluge_cevir(t))
                if yanit.status_code == 200:
                    gonderilen += 1
                else:
                    hatali += 1
    except httpx.RequestError as e:
        return {"ulasildi": False, "sebep": f"{url} yanıt vermiyor ({type(e).__name__}) — toplayıcı çalışmıyor olabilir"}
    return {"ulasildi": True, "gonderilen": gonderilen, "hatali": hatali}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker-host", default="localhost")
    ap.add_argument("--broker-port", type=int, default=1883)
    ap.add_argument("--toplayici-url", default="http://localhost:8700")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--bekleme-sn", type=float, default=2.0, help="MQTT bağlantı denemesi için bekleme")
    a = ap.parse_args()

    if not HAZIR:
        print("HATA: protokol bağımlılıkları eksik.  pip install -e \".[protokol]\"")
        return 1

    tespitler = sentetik_tespitler_uret(a.n)
    print(f"{len(tespitler)} sentetik tespit üretildi "
          f"({sum(t.sinif == 'anomali' for t in tespitler)} anomali)")

    yayinci = TespitYayincisi(broker_host=a.broker_host, broker_port=a.broker_port)
    yayinci.baglan()
    time.sleep(a.bekleme_sn)  # bağlantı/backfill için kısa bekleme

    dogrudan, tamponlu = 0, 0
    for t in tespitler:
        if yayinci.yayinla(t):
            dogrudan += 1
        else:
            tamponlu += 1
    time.sleep(0.3)

    print(f"MQTT [{a.broker_host}:{a.broker_port}] bağlı: {yayinci.baglandi_mi}  "
          f"doğrudan: {dogrudan}  tamponlu: {tamponlu}  tampon boyutu: {yayinci.tampon.boyut()}")
    yayinci.kapat()

    print(f"\nToplayıcı [{a.toplayici_url}] deneniyor...")
    toplayici_sonuc = toplayiciya_gonder(tespitler, a.toplayici_url)
    if toplayici_sonuc["ulasildi"]:
        print(f"  {toplayici_sonuc['gonderilen']} gönderildi, {toplayici_sonuc['hatali']} hata")
    else:
        print(f"  ulaşılamadı: {toplayici_sonuc['sebep']}")

    config.SONUC_DIZINI.mkdir(parents=True, exist_ok=True)
    yol = config.SONUC_DIZINI / "faz5-protokol.md"
    yol.write_text(
        "\n".join(
            [
                "# Faz 5 — protokol demosu\n",
                f"- üretilen tespit: {len(tespitler)} "
                f"({sum(t.sinif == 'anomali' for t in tespitler)} anomali)",
                f"- MQTT broker [{a.broker_host}:{a.broker_port}]: "
                f"{'bağlandı' if yayinci.baglandi_mi else 'ULAŞILAMADI (beklenen — Docker/mosquitto çalışmıyorsa)'}",
                f"- doğrudan yayınlanan: {dogrudan}  ·  yerel tampona yazılan: {tamponlu}",
                f"- toplayıcı [{a.toplayici_url}]: "
                + (
                    f"{toplayici_sonuc['gonderilen']} kayıt alındı"
                    if toplayici_sonuc["ulasildi"]
                    else f"ULAŞILAMADI ({toplayici_sonuc['sebep']})"
                ),
                "\nCanlı doğrulama için:  `docker compose up`  (mosquitto + toplayıcı), "
                "sonra bu betiği tekrar çalıştır — RB-03 (tampon/backfill) davranışını görmek için "
                "önce broker'sız çalıştır (mesajlar tamponlanır), sonra broker'ı başlat ve "
                "`TespitYayincisi`'nin backfill'ini gözlemle.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nyazildi -> {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
