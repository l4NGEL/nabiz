"""Manifest oluştur -> 3 split uygula -> sızıntı kontrolü -> özet + parmak izi yaz.

Gerçek MIMII varsa onu, yoksa data/ham/sentetik/ dizinini kullanır.

Çıktılar (sonuclar/):
  manifest.parquet              tam manifest + bölme sütunları
  split-ozeti.md                bölme x etiket kırılımı
  split-parmak-izleri.json      her stratejinin SHA-256'sı
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nabiz import config  # noqa: E402
from nabiz.cli import utf8_cikti  # noqa: E402
from nabiz.data import pipeline as P  # noqa: E402
from nabiz.data import splits as S  # noqa: E402

utf8_cikti()


def main() -> int:
    config.dizinleri_hazirla()
    man, kaynak = P.manifest_yukle()
    print(
        f"manifest: {len(man)} örnek, kaynak={kaynak}, "
        f"{man['makine_birimi'].nunique()} birim, {man['makine_tipi'].nunique()} tip"
    )

    stratejiler = P.split_stratejileri(man)
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
        f"# Split özeti\n\nKaynak: `{kaynak}` - {len(man)} örnek - "
        f"{man['makine_birimi'].nunique()} birim\n\n" + "\n".join(ozet_bloklari),
        encoding="utf-8",
    )
    print(f"\nyazildi -> {config.SONUC_DIZINI}/  (manifest.parquet, split-ozeti.md, split-parmak-izleri.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
