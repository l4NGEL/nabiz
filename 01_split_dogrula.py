"""Manifest oluştur → 3 split uygula → sızıntı kontrolü → özet + parmak izi yaz.

Gerçek MIMII varsa onu, yoksa data/ham/sentetik/ dizinini kullanır.

Çıktılar (sonuclar/):
  manifest.parquet              tam manifest + üç bölme sütunu
  split-ozeti.md                bölme × etiket kırılımı
  split-parmak-izleri.json      her stratejinin SHA-256'sı (CI bunu doğrular)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import manifest as M  # noqa: E402
from nabiz.data import splits as S  # noqa: E402

utf8_cikti()


def manifest_yukle() -> tuple[pd.DataFrame, str]:
    mimii_kok = config.HAM_VERI / "mimii"
    if mimii_kok.exists() and any(mimii_kok.rglob("*.wav")):
        return M.mimii_manifest(mimii_kok), "mimii"
    sentetik_kok = config.HAM_VERI / "sentetik"
    if sentetik_kok.exists() and any(sentetik_kok.rglob("*.wav")):
        return M.dizin_manifest(sentetik_kok, veri_seti="sentetik"), "sentetik-disk"
    # son çare: dosyasız sentetik manifest
    return M.sentetik_manifest(seed=config.VARSAYILAN_SEED), "sentetik-bellek"


def main() -> int:
    config.dizinleri_hazirla()
    man, kaynak = manifest_yukle()
    print(f"manifest: {len(man)} örnek, kaynak={kaynak}, "
          f"{man['makine_birimi'].nunique()} birim, {man['makine_tipi'].nunique()} tip")

    rejimler = sorted(man["rejim"].unique())
    holdout = [rejimler[-1]] if len(rejimler) > 1 else None

    stratejiler = {
        "birim_ici": S.birim_ici_split(man, seed=config.VARSAYILAN_SEED),
        "birim_disi": S.birim_disi_split(man, seed=config.VARSAYILAN_SEED),
    }
    if holdout:
        stratejiler["rejim_holdout"] = S.rejim_holdout_split(
            man, holdout_rejimler=holdout, seed=config.VARSAYILAN_SEED
        )

    # birleşik manifest: her strateji kendi bölme sütununda
    birlesik = man.copy()
    parmak_izleri: dict[str, str] = {}
    ozet_bloklari: list[str] = []
    for ad, df in stratejiler.items():
        S.dosya_sizintisi_yok(df)
        if ad == "birim_disi":
            S.sizinti_yok(df, "makine_birimi")
        birlesik[f"bolme__{ad}"] = df[S.BOLME_SUTUNU].to_numpy()
        parmak_izleri[ad] = S.manifest_parmak_izi(df)
        ozet_bloklari.append(f"### {ad}\n\n{S.bolme_ozeti(df).to_markdown()}\n")
        print(f"  [{ad}] sizinti yok | parmak izi {parmak_izleri[ad][:12]}...")

    config.SONUC_DIZINI.mkdir(parents=True, exist_ok=True)
    birlesik.to_parquet(config.SONUC_DIZINI / "manifest.parquet", index=False)
    (config.SONUC_DIZINI / "split-parmak-izleri.json").write_text(
        json.dumps({"kaynak": kaynak, "parmak_izleri": parmak_izleri}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.SONUC_DIZINI / "split-ozeti.md").write_text(
        f"# Split özeti\n\nKaynak: `{kaynak}` · {len(man)} örnek · "
        f"{man['makine_birimi'].nunique()} birim\n\n" + "\n".join(ozet_bloklari),
        encoding="utf-8",
    )
    print(f"\nyazildi -> {config.SONUC_DIZINI}/  (manifest.parquet, split-ozeti.md, split-parmak-izleri.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
