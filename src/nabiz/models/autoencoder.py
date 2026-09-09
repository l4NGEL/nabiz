"""Dense autoencoder anomali dedektörü — DCASE Task 2 baseline eşdeğeri.

Girdi: pencere öznitelik vektörü (ölçeklenmiş). Anomali skoru = yeniden
yapılandırma MSE'si. Yalnız normal örneklerle eğitilir.

torch gerektirir (`pip install -e ".[dl]"`). GPU varsa otomatik kullanır.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nabiz.models.temel import dogrula_2b


def _cihaz() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _AGAgi(nn.Module):
    """Simetrik dense autoencoder (DCASE Task 2 baseline tarzı, BatchNorm'suz —
    BN eval-modu anomali skorunu bozabiliyor)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cozucu(self.kodlayici(x))


class DenseAE:
    ad = "dense_ae"

    def __init__(
        self,
        gizli: tuple[int, ...] = (128, 64),
        dar: int = 16,
        epok: int = 120,
        yigin: int = 256,
        ogrenme_hizi: float = 1e-3,
        sabir: int = 12,
        seed: int = 0,
    ) -> None:
        self.gizli = gizli
        self.dar = dar
        self.epok = epok
        self.yigin = yigin
        self.ogrenme_hizi = ogrenme_hizi
        self.sabir = sabir
        self.seed = seed
        self._agi: _AGAgi | None = None
        self.gecmis_: list[float] = []

    def egit(self, X: np.ndarray, X_val: np.ndarray | None = None) -> DenseAE:
        X = dogrula_2b(X)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cihaz = _cihaz()
        self._agi = _AGAgi(X.shape[1], self.gizli, self.dar).to(cihaz)
        opt = torch.optim.Adam(self._agi.parameters(), lr=self.ogrenme_hizi)
        kayip_fn = nn.MSELoss()

        Xt = torch.tensor(X, dtype=torch.float32, device=cihaz)
        val_t = (
            torch.tensor(dogrula_2b(X_val), dtype=torch.float32, device=cihaz)
            if X_val is not None and len(X_val)
            else None
        )
        # küçük veride bile epok başına birkaç gradyan adımı kalsın
        etkin_yigin = max(8, min(self.yigin, len(X) // 8 or len(X)))
        yukleyici = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt),
            batch_size=etkin_yigin,
            shuffle=True,
            drop_last=len(X) >= 2 * etkin_yigin,
        )

        en_iyi, en_iyi_durum, bekleme = np.inf, None, 0
        for _ep in range(self.epok):
            self._agi.train()
            for (yigin,) in yukleyici:
                opt.zero_grad()
                kayip = kayip_fn(self._agi(yigin), yigin)
                kayip.backward()
                opt.step()

            self._agi.eval()
            with torch.no_grad():
                ref = val_t if val_t is not None else Xt
                dv = float(kayip_fn(self._agi(ref), ref))
            self.gecmis_.append(dv)
            if dv < en_iyi - 1e-5:
                en_iyi, bekleme = dv, 0
                en_iyi_durum = {k: v.detach().clone() for k, v in self._agi.state_dict().items()}
            else:
                bekleme += 1
                if bekleme >= self.sabir:
                    break
        if en_iyi_durum is not None:
            self._agi.load_state_dict(en_iyi_durum)
        return self

    def anomali_skoru(self, X: np.ndarray) -> np.ndarray:
        if self._agi is None:
            raise RuntimeError("DenseAE.egit() çağrılmadı")
        X = dogrula_2b(X)
        cihaz = _cihaz()
        self._agi.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=cihaz)
            yeniden = self._agi(Xt)
            return ((yeniden - Xt) ** 2).mean(dim=1).cpu().numpy()
