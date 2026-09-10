"""CRNN sıralı autoencoder — çerçeve projeksiyonu + GRU kodlayıcı/çözücü.

Girdi: (N, n_mel, T). Kodlayıcı GRU son gizli durumu = darboğaz (zamansal
sıkıştırma); çözücü GRU her çerçeveyi yeniden üretir. Anomali skoru = MSE.
"""
from __future__ import annotations

import torch
from torch import nn

from nabiz.models._egitim import TemelAE


class _CRNNAgi(nn.Module):
    def __init__(self, n_mel: int, t: int, gizli: int, gru_kat: int) -> None:
        super().__init__()
        self.t = t
        self.giris_izd = nn.Linear(n_mel, gizli)
        self.kod_gru = nn.GRU(gizli, gizli, gru_kat, batch_first=True)
        self.coz_gru = nn.GRU(gizli, gizli, gru_kat, batch_first=True)
        self.cikis_izd = nn.Linear(gizli, n_mel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.giris_izd(x.transpose(1, 2)))     # (N, T, gizli)
        _, son = self.kod_gru(h)                              # son: (kat, N, gizli)
        tekrar = son[-1].unsqueeze(1).expand(-1, self.t, -1)  # darboğazı T'ye yay
        d, _ = self.coz_gru(tekrar.contiguous(), son.contiguous())
        return self.cikis_izd(d).transpose(1, 2)              # (N, n_mel, T)


class CRNNAE(TemelAE):
    ad = "crnn_ae"

    def __init__(self, gizli: int = 64, gru_kat: int = 1, **kw) -> None:
        kw.setdefault("yigin", 64)
        super().__init__(**kw)
        self.gizli = gizli
        self.gru_kat = gru_kat

    def _agi_kur(self, girdi_sekli: tuple[int, ...]) -> nn.Module:
        n_mel, t = girdi_sekli
        return _CRNNAgi(n_mel, t, self.gizli, self.gru_kat)
