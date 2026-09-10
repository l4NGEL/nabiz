"""Yeniden yapılandırma (autoencoder) modelleri için ortak eğitim çekirdeği.

DenseAE / ConvAE / CRNNAE / SpekTransformerAE hepsi `TemelAE`'yi genişletir:
tek fark ağ mimarisi (`_agi_kur`). Eğitim, erken durma, skorlama burada.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from nabiz.models.temel import dogrula_2b


def cihaz_sec() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dogrula_girdi(X: np.ndarray, ad: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim < 2:
        raise ValueError(f"{ad} en az 2B olmalı (n_ornek, ...), shape={X.shape}")
    if X.size == 0:
        raise ValueError(f"{ad} boş")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{ad} sonlu değil (NaN/inf)")
    return X


def yeniden_yapilandirma_egit(
    agi: nn.Module,
    X: np.ndarray,
    X_val: np.ndarray | None = None,
    *,
    epok: int = 120,
    yigin: int = 128,
    ogrenme_hizi: float = 1e-3,
    sabir: int = 12,
    seed: int = 0,
    cihaz: torch.device | None = None,
) -> tuple[nn.Module, list[float]]:
    cihaz = cihaz or cihaz_sec()
    torch.manual_seed(seed)
    np.random.seed(seed)
    agi = agi.to(cihaz)
    opt = torch.optim.Adam(agi.parameters(), lr=ogrenme_hizi)
    kayip_fn = nn.MSELoss()

    Xt = torch.tensor(_dogrula_girdi(X), device=cihaz)
    val_t = (
        torch.tensor(_dogrula_girdi(X_val, "X_val"), device=cihaz)
        if X_val is not None and len(X_val)
        else None
    )
    etkin = max(8, min(yigin, len(X) // 8 or len(X)))
    yukleyici = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt),
        batch_size=etkin,
        shuffle=True,
        drop_last=len(X) >= 2 * etkin,
    )

    gecmis: list[float] = []
    en_iyi, en_iyi_durum, bekle = float("inf"), None, 0
    for _ep in range(epok):
        agi.train()
        for (yigin_x,) in yukleyici:
            opt.zero_grad()
            kayip_fn(agi(yigin_x), yigin_x).backward()
            opt.step()
        agi.eval()
        with torch.no_grad():
            ref = val_t if val_t is not None else Xt
            dv = float(kayip_fn(agi(ref), ref))
        gecmis.append(dv)
        if dv < en_iyi - 1e-5:
            en_iyi, bekle = dv, 0
            en_iyi_durum = {k: v.detach().clone() for k, v in agi.state_dict().items()}
        else:
            bekle += 1
            if bekle >= sabir:
                break
    if en_iyi_durum is not None:
        agi.load_state_dict(en_iyi_durum)
    return agi, gecmis


def yeniden_yapilandirma_skoru(
    agi: nn.Module, X: np.ndarray, cihaz: torch.device | None = None
) -> np.ndarray:
    cihaz = cihaz or cihaz_sec()
    agi.to(cihaz).eval()
    with torch.no_grad():
        Xt = torch.tensor(_dogrula_girdi(X), device=cihaz)
        yeniden = agi(Xt)
        eksen = tuple(range(1, Xt.dim()))
        return ((yeniden - Xt) ** 2).mean(dim=eksen).cpu().numpy()


class TemelAE:
    """Ortak arayüz: egit / anomali_skoru. Alt sınıf yalnız `_agi_kur` verir."""

    ad = "temel_ae"

    def __init__(
        self,
        epok: int = 120,
        yigin: int = 128,
        ogrenme_hizi: float = 1e-3,
        sabir: int = 12,
        seed: int = 0,
    ) -> None:
        self._egitim_kw: dict[str, Any] = {
            "epok": epok,
            "yigin": yigin,
            "ogrenme_hizi": ogrenme_hizi,
            "sabir": sabir,
            "seed": seed,
        }
        self._agi: nn.Module | None = None
        self.gecmis_: list[float] = []

    def _agi_kur(self, girdi_sekli: tuple[int, ...]) -> nn.Module:
        raise NotImplementedError

    def egit(self, X: np.ndarray, X_val: np.ndarray | None = None) -> TemelAE:
        girdi = _dogrula_girdi(X)
        # ağ kurulmadan ÖNCE tohumla — nn.Linear/Conv ağırlık başlangıcı global RNG kullanır
        torch.manual_seed(self._egitim_kw["seed"])
        np.random.seed(self._egitim_kw["seed"])
        self._agi = self._agi_kur(girdi.shape[1:])
        self._agi, self.gecmis_ = yeniden_yapilandirma_egit(
            self._agi, girdi, X_val, **self._egitim_kw
        )
        return self

    def anomali_skoru(self, X: np.ndarray) -> np.ndarray:
        if self._agi is None:
            raise RuntimeError(f"{self.ad}.egit() çağrılmadı")
        return yeniden_yapilandirma_skoru(self._agi, X)


# DenseAE hâlâ 2B tablo öznitelik alır — dogrula_2b'yi burada da kullanışa aç
__all__ = [
    "TemelAE",
    "cihaz_sec",
    "dogrula_2b",
    "yeniden_yapilandirma_egit",
    "yeniden_yapilandirma_skoru",
]
