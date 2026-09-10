"""Spektrogram Transformer autoencoder — maskeli çerçeve yeniden yapılandırma.

Her zaman çerçevesi bir token (n_mel boyut). Eğitimde çerçevelerin bir kısmı
maskelenir (MAE tarzı); model tam diziyi yeniden üretmeyi öğrenir. Anomali
skoru = maskesiz girdide yeniden yapılandırma MSE'si.
"""
from __future__ import annotations

import torch
from torch import nn

from nabiz.models._egitim import TemelAE


class _SpekTransformerAgi(nn.Module):
    def __init__(
        self, n_mel: int, t: int, d_model: int, kafa: int, kat: int, maske_orani: float
    ) -> None:
        super().__init__()
        self.maske_orani = maske_orani
        self.giris = nn.Linear(n_mel, d_model)
        self.poz = nn.Parameter(torch.randn(1, t, d_model) * 0.02)
        katman = nn.TransformerEncoderLayer(
            d_model, kafa, dim_feedforward=2 * d_model, batch_first=True, dropout=0.0
        )
        self.enkoder = nn.TransformerEncoder(katman, kat)
        self.cikis = nn.Linear(d_model, n_mel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.giris(x.transpose(1, 2)) + self.poz          # (N, T, d_model)
        if self.training and self.maske_orani > 0:
            m = torch.rand(h.shape[:2], device=h.device) < self.maske_orani
            h = h.masked_fill(m.unsqueeze(-1), 0.0)
        h = self.enkoder(h)
        return self.cikis(h).transpose(1, 2)                  # (N, n_mel, T)


class SpekTransformerAE(TemelAE):
    ad = "spek_transformer_ae"

    def __init__(
        self,
        d_model: int = 64,
        kafa: int = 4,
        kat: int = 2,
        maske_orani: float = 0.5,
        **kw,
    ) -> None:
        kw.setdefault("yigin", 64)
        kw.setdefault("ogrenme_hizi", 5e-4)
        super().__init__(**kw)
        if d_model % kafa:
            raise ValueError(f"d_model ({d_model}) kafa ({kafa}) ile bölünmeli")
        self.d_model, self.kafa, self.kat, self.maske_orani = d_model, kafa, kat, maske_orani

    def _agi_kur(self, girdi_sekli: tuple[int, ...]) -> nn.Module:
        n_mel, t = girdi_sekli
        return _SpekTransformerAgi(n_mel, t, self.d_model, self.kafa, self.kat, self.maske_orani)
