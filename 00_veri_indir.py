"""Veri setlerini indir veya sentetik geliştirme verisi üret.

Kullanım:
  python 00_veri_indir.py --sentetik            # offline: küçük sahte WAV seti
  python 00_veri_indir.py --mimii-bilgi         # MIMII indirme talimatları
  python 00_veri_indir.py --mimii fan --snr -6  # tek makine tipini indir (büyük)
  python 00_veri_indir.py --cmapss-sentetik     # offline: sahte C-MAPSS trajectory'leri
  python 00_veri_indir.py --cmapss-bilgi        # C-MAPSS indirme talimatları

Sentetik veri MIMII dizin düzenini taklit eder:
  data/ham/sentetik/<tip>/id_<NN>/<normal|abnormal>/<dosya>.wav
böylece tüm boru hattı (manifest → split → öznitelik) gerçek veri olmadan çalışır.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import cmapss as C  # noqa: E402

utf8_cikti()

MIMII_ZENODO = "https://zenodo.org/records/3384388"
MIMII_SNR = {"-6": "-6_dB", "0": "0_dB", "6": "6_dB", "min6dB": "-6_dB"}
CMAPSS_KAYNAK = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"


def sentetik_wav_uret(
    kok: Path,
    n_birim: int = 4,
    n_normal: int = 40,
    n_anomali: int = 8,
    sr: int = 16_000,
    sure_sn: float = 2.0,
    seed: int = 0,
) -> int:
    """MIMII benzeri düzende küçük sahte WAV seti yaz. Dosya sayısını döner.

    Düzen:  <kok>/sentetik/<snr>/<tip>/id_<NN>/<normal|abnormal>/<dosya>.wav
    İki 'rejim' (SNR koşulu) → tüm boru hattı (birim-içi/dışı + rejim-holdout)
    gerçek veri olmadan çalışır.
    """
    import soundfile as sf

    rng = np.random.default_rng(seed)
    yazilan = 0
    t = np.arange(int(sure_sn * sr)) / sr
    for snr_ad, gurultu in (("6dB", 0.02), ("0dB", 0.06)):
        for tip, taban_f in (("fan", 100.0), ("pump", 160.0)):
            for b in range(n_birim):
                birim_kayma = rng.uniform(-5, 5)  # üniteye özgü frekans kayması
                for durum, adet in (("normal", n_normal), ("abnormal", n_anomali)):
                    d = kok / "sentetik" / snr_ad / tip / f"id_{b:02d}" / durum
                    d.mkdir(parents=True, exist_ok=True)
                    for i in range(adet):
                        f0 = taban_f + birim_kayma
                        x = 0.30 * np.sin(2 * np.pi * f0 * t)
                        x += 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
                        x += gurultu * rng.standard_normal(t.size)
                        if durum == "abnormal":
                            # geçici darbeler (rulman arıza benzeri) + yan bant
                            for _ in range(int(rng.integers(3, 8))):
                                p = int(rng.integers(0, t.size - 60))
                                x[p : p + 50] += rng.uniform(0.3, 0.8) * np.hanning(50)
                            x += 0.10 * np.sin(2 * np.pi * (f0 + 37.0) * t)
                        x = (x / np.max(np.abs(x)) * 0.9).astype(np.float32)
                        sf.write(d / f"{i:06d}.wav", x, sr)
                        yazilan += 1
    return yazilan


def mimii_bilgi() -> None:
    print(
        f"""
MIMII veri seti — akustik endüstriyel makine anomali tespiti (DCASE 2020 Task 2 tabanı)

  Kaynak : {MIMII_ZENODO}
  Lisans : CC BY 4.0
  Boyut  : 12 dosya, toplam ~100 GB — HER (makine tipi × SNR) AYRI zip, ~7-11 GB
           (Zenodo API'den doğrulandı 2026-09-15; önceki '~10 GB hepsi' notu YANLIŞTI)
           fan ~10.2-10.9 GB · pump ~7.7-8.2 GB · slider ~7.1-8.0 GB · valve ~6.9-8.0 GB

İndirme (örnek — fan+pump, 6dB+0dB, ~36 GB — makine tipi kırılımı + SNR domain-shift için yeterli):
  mkdir -p {config.HAM_VERI}/mimii/6dB {config.HAM_VERI}/mimii/0dB
  cd {config.HAM_VERI}
  curl -LO {MIMII_ZENODO}/files/6_dB_fan.zip
  curl -LO {MIMII_ZENODO}/files/6_dB_pump.zip
  curl -LO {MIMII_ZENODO}/files/0_dB_fan.zip
  curl -LO {MIMII_ZENODO}/files/0_dB_pump.zip
  unzip 6_dB_fan.zip 6_dB_pump.zip -d mimii/6dB && unzip 0_dB_fan.zip 0_dB_pump.zip -d mimii/0dB
  # zip içi üst klasör doğrudan <tip>/ ise (fan/id_00/...) yukarıdaki -d hedefi doğru;
  # farklıysa mimii/<snr>/ altına <tip>/id_NN/<durum>/*.wav gelecek şekilde taşı

Beklenen düzen:
  {config.HAM_VERI}/mimii/<snr>/<tip>/id_<NN>/<normal|abnormal>/*.wav

Sonra:  python 01_split_dogrula.py
"""
    )


def cmapss_bilgi() -> None:
    print(
        f"""
C-MAPSS veri seti — NASA türbofan motor bozulma simülasyonu (RUL tahmini)

  Kaynak : {CMAPSS_KAYNAK}
  Lisans : kamu malı (NASA)
  Dosya  : train_FD001.txt, test_FD001.txt, RUL_FD001.txt (4 alt-set: FD001-FD004)

İndirdikten sonra:
  {config.HAM_VERI}/cmapss/train_FD001.txt
  {config.HAM_VERI}/cmapss/test_FD001.txt
  {config.HAM_VERI}/cmapss/RUL_FD001.txt

Sonra:  python 05_rul_egit.py
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentetik", action="store_true", help="küçük sahte WAV seti üret")
    ap.add_argument("--mimii-bilgi", action="store_true", help="MIMII indirme talimatları")
    ap.add_argument("--mimii", metavar="TIP", help="MIMII makine tipi (fan/pump/slider/valve)")
    ap.add_argument("--snr", default="-6", help="MIMII SNR koşulu (-6 / 0 / 6)")
    ap.add_argument("--cmapss-sentetik", action="store_true", help="sahte C-MAPSS trajectory'leri üret")
    ap.add_argument("--cmapss-bilgi", action="store_true", help="C-MAPSS indirme talimatları")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    config.dizinleri_hazirla()

    if a.mimii_bilgi:
        mimii_bilgi()
        return 0

    if a.cmapss_bilgi:
        cmapss_bilgi()
        return 0

    if a.cmapss_sentetik:
        df = C.sentetik_uret(seed=a.seed)
        hedef = config.HAM_VERI / "cmapss" / "train_sentetik.csv"
        hedef.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(hedef, index=False)
        print(f"{df['birim'].nunique()} birim, {len(df)} satır yazildi -> {hedef}")
        return 0

    if a.sentetik:
        n = sentetik_wav_uret(config.HAM_VERI, seed=a.seed)
        print(f"{n} sentetik WAV yazildi -> {config.HAM_VERI / 'sentetik'}")
        return 0

    if a.mimii:
        print(
            "Otomatik MIMII indirme henüz bağlanmadı (dosyalar büyük). "
            "'python 00_veri_indir.py --mimii-bilgi' ile manuel adımlar."
        )
        return 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
