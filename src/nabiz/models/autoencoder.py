"""Dense autoencoder anomali dedektörü — DCASE Task 2 baseline eşdeğeri.

Girdi: pencere öznitelik vektörü (ölçeklenmiş, 2B tablo). Anomali skoru =
yeniden yapılandırma MSE'si. Yalnız normal örneklerle eğitilir.

torch gerektirir (`pip install -e ".[dl]"`). GPU varsa otomatik kullanır.
Eğitim/skorlama çekirdeği: `nabiz.models._egitim`.
"""
from __future__ import annotations

from torch import nn

from nabiz.models._egitim import TemelAE


class _DenseAgi(nn.Module):
    """Simetrik dense autoencoder (BatchNorm'suz — BN eval-modu anomali
    skorunu bozabiliyor)."""

    def __init__(self, girdi: int, gizli: tuple[int, ...], dar: int) -> None:
        super().__init__()
        kod: list[nn.Module] = []
        onceki = girdi
        for h in gizli:
            kod += [nn.Linear(onceki, h), nn.ReLU()]
            onceki = h
        kod.append(nn.Linear(onceki, dar))
        self.kodlayici = nn.Sequential(*kod)

        coz: list[nn.Module] = []
        onceki = dar
        for h in reversed(gizli):
            coz += [nn.Linear(onceki, h), nn.ReLU()]
            onceki = h
        coz.append(nn.Linear(onceki, girdi))
        self.cozucu = nn.Sequential(*coz)

    def forward(self, x):
        return self.cozucu(self.kodlayici(x))


class DenseAE(TemelAE):
    ad = "dense_ae"

    def __init__(
        self,
        gizli: tuple[int, ...] = (128, 64),
        dar: int = 16,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.gizli = gizli
        self.dar = dar

    def _agi_kur(self, girdi_sekli: tuple[int, ...]) -> nn.Module:
        (d,) = girdi_sekli
        return _DenseAgi(d, self.gizli, self.dar)
