"""Yerel tampon (WAL) — broker ulaşılamazken mesajları biriktirir (RB-03).

SQLite tabanlı, süreç yeniden başlasa da kalıcı (dosya yolu verilirse).
`yayinci.py` bunu sarar; burada saf depolama mantığı — MQTT'ye bağımlılık yok,
bu yüzden broker olmadan tam test edilebilir.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class YerelTampon:
    def __init__(self, yol: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(yol), check_same_thread=False)
        self._kilit = threading.Lock()
        with self._kilit:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS kuyruk (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mesaj_id TEXT NOT NULL,
                    veri BLOB NOT NULL,
                    eklenme_zamani REAL NOT NULL
                )"""
            )
            self._db.commit()

    def ekle(self, mesaj_id: str, veri: bytes) -> int:
        with self._kilit:
            imlec = self._db.execute(
                "INSERT INTO kuyruk (mesaj_id, veri, eklenme_zamani) VALUES (?, ?, ?)",
                (mesaj_id, veri, time.time()),
            )
            self._db.commit()
            return int(imlec.lastrowid)

    def hepsi(self) -> list[tuple[int, str, bytes]]:
        with self._kilit:
            return list(self._db.execute("SELECT id, mesaj_id, veri FROM kuyruk ORDER BY id"))

    def sil(self, id_: int) -> None:
        with self._kilit:
            self._db.execute("DELETE FROM kuyruk WHERE id = ?", (id_,))
            self._db.commit()

    def boyut(self) -> int:
        with self._kilit:
            (n,) = self._db.execute("SELECT COUNT(*) FROM kuyruk").fetchone()
            return int(n)

    def temizle(self) -> None:
        with self._kilit:
            self._db.execute("DELETE FROM kuyruk")
            self._db.commit()

    def kapat(self) -> None:
        with self._kilit:
            self._db.close()
