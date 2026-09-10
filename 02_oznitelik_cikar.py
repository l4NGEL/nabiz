"""Öznitelik çıkarım hattı: her WAV -> 2 sn pencereler -> pencere başına öznitelik.

Öznitelikler:
  a_*   akustik: log-mel bant ort/std (2x64), RMS, kurtosis
  z_*   zarf spektrumu: darbe/modülasyon (rulman imzası)
  t_*   telemetri: RPM/yük/sıcaklık/gerilim ort/std/min/maks/aralık/eğim/son/delta
        (Faz 1 vekil — gerçek telemetri yok, dosya-tohumlu sentetik)

Çıktı:  data/islenmis/oznitelikler.parquet   (pencere başına 1 satır)
        sonuclar/oznitelik-ozeti.md
        --mel ile: data/islenmis/mel.npz     (X: (N, n_mel, T) float32, parquet ile aynı satır sırası)

Kullanım:
  python 02_oznitelik_cikar.py               # tüm manifest
  python 02_oznitelik_cikar.py --limit 40    # ilk 40 dosya (hızlı deneme)
  python 02_oznitelik_cikar.py --mel         # log-mel tensör önbelleği de yaz (CNN/CRNN/Transformer)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import manifest as M  # noqa: E402
from nabiz.data import pipeline as P  # noqa: E402
from nabiz.dsp import features as F  # noqa: E402
from nabiz.dsp.telemetri import sentetik_telemetri, telemetri_oznitelik  # noqa: E402

utf8_cikti()

META_SUTUNLAR = ["dosya", "makine_tipi", "makine_birimi", "etiket", "ariza_tipi", "rejim"]


def _dosya_seed(dosya: str) -> int:
    return int(hashlib.sha1(dosya.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def _dosya_pencereleri(yol: Path) -> tuple[np.ndarray, int]:
    veri, sr = sf.read(str(yol), dtype="float64", always_2d=False)
    if veri.ndim > 1:
        veri = veri.mean(axis=1)
    cu = int(config.PENCERE_SN * sr)
    hop = int(config.BINDIRME_SN * sr)
    return F.cerceve(veri, cu, hop), sr


def _pencere_telemetri(tel: dict, bas_sn: float, bitis_sn: float) -> dict:
    z = tel["_zaman_sn"]
    m = (z >= bas_sn) & (z <= bitis_sn)
    if not m.any():  # pencere telemetri örneği içermiyorsa en yakını al
        m = np.zeros_like(z, dtype=bool)
        m[int(np.argmin(np.abs(z - 0.5 * (bas_sn + bitis_sn))))] = True
    return {k: v[m] for k, v in tel.items() if k != "_zaman_sn"}


def satirlari_uret(
    man: pd.DataFrame, limit: int | None, mel: bool
) -> tuple[list[dict], list[np.ndarray] | None]:
    yollar = M.mutlak_yollar(man)
    bolme_sut = [c for c in man.columns if c.startswith("bolme__")]
    satirlar: list[dict] = []
    mel_listesi: list[np.ndarray] | None = [] if mel else None
    hedef_t = F.varsayilan_cerceve_sayisi()
    hatali = 0

    alt = man.head(limit) if limit else man
    for i, (_, satir) in enumerate(alt.iterrows()):
        yol = yollar[i]
        try:
            pencereler, sr = _dosya_pencereleri(yol)
        except Exception as e:  # noqa: BLE001 — bozuk dosya raporlanır, hat durmaz
            hatali += 1
            if hatali <= 3:
                print(f"  ! okunamadı: {yol.name} ({e})")
            continue

        anomali = satir["etiket"] == "anomali"
        sure_sn = (pencereler.shape[0] - 1) * config.BINDIRME_SN + config.PENCERE_SN
        tel = sentetik_telemetri(
            sure_sn=sure_sn, seed=_dosya_seed(satir["dosya"]), anomali=anomali
        )
        for p_idx, pencere in enumerate(pencereler):
            bas = p_idx * config.BINDIRME_SN
            oz = F.pencere_oznitelikleri(pencere, sr)
            oz.update(telemetri_oznitelik(_pencere_telemetri(tel, bas, bas + config.PENCERE_SN)))
            oz.update({k: satir[k] for k in META_SUTUNLAR})
            oz.update({k: satir[k] for k in bolme_sut})
            oz["pencere_idx"] = p_idx
            satirlar.append(oz)
            if mel_listesi is not None:
                mel_listesi.append(
                    F.mel_tensor(pencere, sr, hedef_cerceve=hedef_t).astype(np.float32)
                )

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(alt)} dosya işlendi")

    if hatali:
        print(f"  toplam {hatali} dosya okunamadı")
    return satirlar, mel_listesi


def ozet_yaz(df: pd.DataFrame, kaynak: str, saniye: float, mel_sekil: tuple | None) -> None:
    oz_sut = [c for c in df.columns if c[:2] in ("a_", "z_", "t_")]
    bolme_sut = [c for c in df.columns if c.startswith("bolme__")]
    bloklar = [
        "# Öznitelik özeti\n",
        f"- kaynak: `{kaynak}`",
        f"- satır (pencere): **{len(df)}**  ·  öznitelik: **{len(oz_sut)}** "
        f"(a_ {sum(c.startswith('a_') for c in oz_sut)}, "
        f"z_ {sum(c.startswith('z_') for c in oz_sut)}, "
        f"t_ {sum(c.startswith('t_') for c in oz_sut)})",
        f"- mel tensör: {mel_sekil if mel_sekil else 'yazılmadı (--mel ile)'}",
        f"- süre: {saniye:.1f} sn\n",
    ]
    for b in bolme_sut:
        tab = df.groupby([b, "etiket"]).size().unstack(fill_value=0)
        bloklar.append(f"### {b}\n\n{tab.to_markdown()}\n")
    (config.SONUC_DIZINI / "oznitelik-ozeti.md").write_text("\n".join(bloklar), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="ilk N dosya (hızlı deneme)")
    ap.add_argument("--mel", action="store_true", help="log-mel tensör önbelleği de yaz")
    a = ap.parse_args()

    config.dizinleri_hazirla()
    man, kaynak = P.manifest_yukle()
    if kaynak == "sentetik-bellek":
        print("HATA: gerçek dosya yok. Önce:  python 00_veri_indir.py --sentetik")
        return 1

    man, _parmak = P.splitli_manifest(man)
    print(f"manifest: {len(man)} dosya, kaynak={kaynak}")

    t0 = time.perf_counter()
    satirlar, mel_listesi = satirlari_uret(man, a.limit, a.mel)
    saniye = time.perf_counter() - t0
    if not satirlar:
        print("HATA: hiç öznitelik üretilemedi")
        return 1

    df = pd.DataFrame(satirlar)
    oz_sut = sorted(c for c in df.columns if c[:2] in ("a_", "z_", "t_"))
    df = df[META_SUTUNLAR + [c for c in df.columns if c.startswith("bolme__")] + ["pencere_idx"] + oz_sut]

    config.ISLENMIS_VERI.mkdir(parents=True, exist_ok=True)
    cikti = config.ISLENMIS_VERI / "oznitelikler.parquet"
    df.to_parquet(cikti, index=False)

    mel_sekil = None
    if mel_listesi is not None:
        X_mel = np.stack(mel_listesi)  # (N, n_mel, T)
        mel_sekil = X_mel.shape
        np.savez_compressed(config.ISLENMIS_VERI / "mel.npz", X=X_mel)
        print(f"mel tensör: {X_mel.shape} -> {config.ISLENMIS_VERI / 'mel.npz'}")

    ozet_yaz(df, kaynak, saniye, mel_sekil)
    print(f"\n{len(df)} pencere x {len(oz_sut)} öznitelik  ({saniye:.1f} sn)")
    print(f"yazildi -> {cikti}")
    print(f"yazildi -> {config.SONUC_DIZINI / 'oznitelik-ozeti.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
