"""Ürün dedektörünü görülmemiş MIMII ünitelerinde çalıştır, alarmları panele yaz.

Eşik validasyondan gelir. Paneldeki sınıf, etikete göre değil, kurala göre
konur: bir kayıt klasöründe k ardışık dosya eşiği geçince bir alarm.

  uvicorn nabiz.serving.toplayici:app --app-dir src --port 8701
  python 09_canli_oynat.py

Panel: http://127.0.0.1:8701/
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import oznitelik as OZ  # noqa: E402
from nabiz.runtime.urun import AkustikUrun  # noqa: E402

utf8_cikti()


def _ayakta(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/saglik", timeout=2) as yanit:
            return yanit.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _toplayiciyi_ac(url: str, port: int) -> None:
    if _ayakta(url):
        return
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "nabiz.serving.toplayici:app",
         "--app-dir", "src", "--port", str(port)],
        cwd=str(config.KOK),
    )
    for _ in range(40):
        if _ayakta(url):
            return
        time.sleep(0.25)
    raise RuntimeError(f"toplayıcı açılmadı: {url}")


def _temizle() -> None:
    yol = config.VERI_KOK / "toplayici.db"
    if not yol.exists():
        return
    db = sqlite3.connect(yol)
    db.execute("DELETE FROM tespitler")
    db.commit()
    db.close()


def _gonder(url: str, govde: dict) -> None:
    istek = urllib.request.Request(
        f"{url}/tespit",
        data=json.dumps(govde).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(istek, timeout=5) as yanit:
        if yanit.status != 200:
            raise RuntimeError(f"HTTP {yanit.status}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8701)
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    if not (config.ISLENMIS_VERI / "oznitelikler.parquet").exists():
        print("HATA: öznitelik yok. Önce:  python 02_oznitelik_cikar.py")
        return 1

    print("akustik ürün eğitiliyor (birim_disi, yalnız a_)")
    veri = OZ.yukle("birim_disi", gruplar=("a",))
    urun = AkustikUrun().egit(veri)
    isletme = urun.isletme
    print(f"  işletme: z≥{isletme['esik']}  k={isletme['k']}  val recall {isletme['val_recall']:.3f}")
    olaylar = urun.olaylar(veri)
    normal_olay = sum(1 for o in olaylar if "normal" in set(str(o["oturum"]).replace("\\", "/").split("/")))
    kayitlar = urun.kayitlar(veri)
    print(
        f"  {len(olaylar)} alarm olayı, bunun {normal_olay} tanesi normal kayıt klasöründe"
        f"  ·  {len(kayitlar)} dosyanın tamamı panele yazılıyor"
    )

    _toplayiciyi_ac(url, a.port)
    _temizle()
    model = f"urun-{urun.model.ad}-z{isletme['esik']}-k{isletme['k']}"
    taban = 1_700_000_000_000
    for i, kayit in enumerate(kayitlar):
        if kayit["alarm"]:
            sinif, siddet = "anomali", "ileri"
        elif kayit["sicak"]:
            sinif, siddet = "anomali", "orta"
        else:
            sinif, siddet = "normal", "yok"
        _gonder(url, {
            "mesaj_id": str(uuid.uuid4()),
            "zaman_damgasi_ms": taban + i * 1_000,
            "dugum_id": kayit["klasor"],
            "makine_id": kayit["birim"],
            "anomali_skoru": kayit["skor"],
            "sinif": sinif,
            "guven": float(min(1.0, kayit["skor"] / 8.0)),
            "siddet": siddet,
            "oznitelik_anlik_goruntu": [],
            "model_surumu": model,
            "kaynak_split": "birim_disi",
            "kayit_adi": kayit["dosya"],
        })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(kayitlar)}", flush=True)
    print(f"yazıldı → {url}/  ({len(kayitlar)} kayıt, {len(olaylar)} alarm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
