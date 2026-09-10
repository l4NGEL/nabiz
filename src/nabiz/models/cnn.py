"""2B evrişimli autoencoder — log-mel spektrogram (n_mel, T) üzerinde anomali.

Girdi: (N, n_mel, T). İçeride kanal ekseni eklenir. Anomali skoru = yeniden
yapılandırma MSE'si. `nabiz.data.mel.yukle` ile beslenir.
"""
from __future__ import annotations

import torch
import torch.nn.functional as tf
from torch import nn

from nabiz.models._egitim import TemelAE


def _kucul(n: int, kat: int = 3) -> int:
    for _ in range(kat):
        n = (n + 1) // 2
    return n


class _EvrisimAE(nn.Module):
    def __init__(self, n_mel: int, t: int, kanallar: tuple[int, int, int], dar: int) -> None:
        super().__init__()
        self.n_mel, self.t = n_mel, t
        self.h8, self.w8 = _kucul(n_mel), _kucul(t)
        c1, c2, c3 = kanallar

        self.kodlayici = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dar_ileri = nn.Linear(c3 * self.h8 * self.w8, dar)
        self.dar_geri = nn.Linear(dar, c3 * self.h8 * self.w8)
        self._c3 = c3
        self.cozucu = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.kodlayici(x.unsqueeze(1))                       # (N, c3, h8, w8)
        z = self.dar_geri(self.dar_ileri(z.flatten(1)))
        z = z.view(-1, self._c3, self.h8, self.w8)
        y = self.cozucu(z)                                       # (N, 1, ~8*h8, ~8*w8)
        y = y[..., : self.n_mel, : self.t]
        if y.shape[-2:] != (self.n_mel, self.t):
            y = tf.pad(y, (0, self.t - y.shape[-1], 0, self.n_mel - y.shape[-2]))
        return y.squeeze(1)


class ConvAE(TemelAE):
    ad = "conv_ae"

    def __init__(
        self,
        kanallar: tuple[int, int, int] = (16, 32, 64),
        dar: int = 32,
        **kw,
    ) -> None:
        kw.setdefault("yigin", 64)
        super().__init__(**kw)
        self.kanallar = kanallar
        self.dar = dar

    def _agi_kur(self, girdi_sekli: tuple[int, ...]) -> nn.Module:
        n_mel, t = girdi_sekli
        return _EvrisimAE(n_mel, t, self.kanallar, self.dar)
