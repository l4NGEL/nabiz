"""Numaralı boru hattı betikleri için ortak yardımcılar."""
from __future__ import annotations

import sys


def utf8_cikti() -> None:
    """stdout/stderr'i UTF-8'e çevir.

    Windows Türkçe yerelinde konsol cp1254; '→' '…' gibi karakterler
    UnicodeEncodeError verir. 3.7+ reconfigure ile çözülür.
    """
    for akis in (sys.stdout, sys.stderr):
        try:
            akis.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
