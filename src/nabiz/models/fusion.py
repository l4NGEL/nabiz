"""Füzyon autoencoder — spektrogram-CNN ⊕ telemetri-MLP ortak darboğaz (hibrid).

İki girdi: log-mel tensörü (n_mel, T) ve telemetri öznitelik vektörü (n_tel,).
Ayrı kodlayıcılar ortak bir darboğazda birleşir; ayrı çözücüler her dalı geri
üretir. Anomali skoru = ağırlıklı toplam yeniden yapılandırma hatası.

İlandaki "hibrid model" maddesinin karşılığı — titreşim/akustik + telemetriyi
tek modelde birleştiren tek mimari (docs/veri-toplama-sartnamesi.md §8).
`nabiz.models._egitim.TemelAE`'yi genişletmez: iki girdi aldığı için kendi
eğitim döngüsü var, ama aynı desenler (ağ-öncesi tohumlama, erken durma,
küçük-veri batch boyutu) tekrarlanır.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as tf
from torch import nn

from nabiz.models._egitim import cihaz_sec
from nabiz.models.cnn import _kucul


def _dogrula(X: np.ndarray, ad: str, ndim: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != ndim:
        raise ValueError(f"{ad} {ndim}B olmalı, shape={X.shape}")
    if X.size == 0:
        raise ValueError(f"{ad} boş")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{ad} sonlu değil (NaN/inf)")
    return X


class _SpekKodlayici(nn.Module):
    def __init__(self, n_mel: int, t: int, kanallar: tuple[int, int, int], dar: int) -> None:
        super().__init__()
        self.h8, self.w8 = _kucul(n_mel), _kucul(t)
        c1, c2, c3 = kanallar
        self._c3 = c3
        self.gov = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.duz = nn.Linear(c3 * self.h8 * self.w8, dar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.duz(self.gov(x.unsqueeze(1)).flatten(1))


class _SpekCozucu(nn.Module):
    def __init__(self, n_mel: int, t: int, kanallar: tuple[int, int, int], dar: int) -> None:
        super().__init__()
        self.n_mel, self.t = n_mel, t
        self.h8, self.w8 = _kucul(n_mel), _kucul(t)
        c1, c2, c3 = kanallar
        self._c3 = c3
        self.genislet = nn.Linear(dar, c3 * self.h8 * self.w8)
        self.gov = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.genislet(z).view(-1, self._c3, self.h8, self.w8)
        y = self.gov(x)[..., : self.n_mel, : self.t]
        if y.shape[-2:] != (self.n_mel, self.t):
            y = tf.pad(y, (0, self.t - y.shape[-1], 0, self.n_mel - y.shape[-2]))
        return y.squeeze(1)


def _mlp(katmanlar: list[int]) -> nn.Sequential:
    parcalar: list[nn.Module] = []
    for a, b in zip(katmanlar[:-1], katmanlar[1:], strict=True):
        parcalar += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*parcalar[:-1])  # son ReLU'yu at — doğrusal çıkış


class _FuzyonAgi(nn.Module):
    def __init__(
        self,
        n_mel: int, t: int, n_tel: int,
        spek_kanallar: tuple[int, int, int], spek_dar: int,
        tel_gizli: tuple[int, ...], tel_dar: int,
        ortak_dar: int,
    ) -> None:
        super().__init__()
        self.spek_kod = _SpekKodlayici(n_mel, t, spek_kanallar, spek_dar)
        self.spek_coz = _SpekCozucu(n_mel, t, spek_kanallar, spek_dar)
        self.tel_kod = _mlp([n_tel, *tel_gizli, tel_dar])
        self.tel_coz = _mlp([tel_dar, *reversed(tel_gizli), n_tel])
        self.ortak_ileri = nn.Sequential(nn.Linear(spek_dar + tel_dar, ortak_dar), nn.ReLU())
        self.ortak_geri = nn.Sequential(nn.Linear(ortak_dar, spek_dar + tel_dar), nn.ReLU())
        self.spek_dar, self.tel_dar = spek_dar, tel_dar

    def forward(self, x_spek: torch.Tensor, x_tel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.ortak_ileri(torch.cat([self.spek_kod(x_spek), self.tel_kod(x_tel)], dim=1))
        z2 = self.ortak_geri(z)
        z_spek, z_tel = z2[:, : self.spek_dar], z2[:, self.spek_dar :]
        return self.spek_coz(z_spek), self.tel_coz(z_tel)


class FusionAE:
    """İki girdili hibrid autoencoder. `AnomaliModeli` protokolüne UYMAZ
    (egit/anomali_skoru iki dizi alır) — 03_model_egit.py ayrı ele alır."""

    ad = "fusion_ae"

    def __init__(
        self,
        spek_kanallar: tuple[int, int, int] = (16, 32, 64),
        spek_dar: int = 32,
        tel_gizli: tuple[int, ...] = (32, 16),
        tel_dar: int = 8,
        ortak_dar: int = 16,
        tel_agirlik: float = 0.3,
        epok: int = 100,
        yigin: int = 64,
        ogrenme_hizi: float = 1e-3,
        sabir: int = 12,
        seed: int = 0,
    ) -> None:
        self.spek_kanallar, self.spek_dar = spek_kanallar, spek_dar
        self.tel_gizli, self.tel_dar, self.ortak_dar = tel_gizli, tel_dar, ortak_dar
        self.tel_agirlik = tel_agirlik
        self.epok, self.yigin = epok, yigin
        self.ogrenme_hizi, self.sabir, self.seed = ogrenme_hizi, sabir, seed
        self._agi: _FuzyonAgi | None = None
        self.gecmis_: list[float] = []

    def egit(
        self,
        X_spek: np.ndarray,
        X_tel: np.ndarray,
        X_spek_val: np.ndarray | None = None,
        X_tel_val: np.ndarray | None = None,
    ) -> FusionAE:
        X_spek, X_tel = _dogrula(X_spek, "X_spek", 3), _dogrula(X_tel, "X_tel", 2)
        if len(X_spek) != len(X_tel):
            raise ValueError(f"X_spek ({len(X_spek)}) ve X_tel ({len(X_tel)}) örnek sayısı uyuşmuyor")

        cihaz = cihaz_sec()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n_mel, t = X_spek.shape[1], X_spek.shape[2]
        self._agi = _FuzyonAgi(
            n_mel, t, X_tel.shape[1], self.spek_kanallar, self.spek_dar,
            self.tel_gizli, self.tel_dar, self.ortak_dar,
        ).to(cihaz)
        opt = torch.optim.Adam(self._agi.parameters(), lr=self.ogrenme_hizi)
        mse = nn.MSELoss()

        Xs = torch.tensor(X_spek, device=cihaz)
        Xt = torch.tensor(X_tel, device=cihaz)
        val = None
        if X_spek_val is not None and X_tel_val is not None and len(X_spek_val):
            val = (
                torch.tensor(_dogrula(X_spek_val, "X_spek_val", 3), device=cihaz),
                torch.tensor(_dogrula(X_tel_val, "X_tel_val", 2), device=cihaz),
            )

        etkin = max(8, min(self.yigin, len(Xs) // 8 or len(Xs)))
        yukleyici = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xs, Xt),
            batch_size=etkin, shuffle=True, drop_last=len(Xs) >= 2 * etkin,
        )

        en_iyi, en_iyi_durum, bekle = float("inf"), None, 0
        for _ep in range(self.epok):
            self._agi.train()
            for bs, bt in yukleyici:
                opt.zero_grad()
                ys, yt = self._agi(bs, bt)
                kayip = mse(ys, bs) + self.tel_agirlik * mse(yt, bt)
                kayip.backward()
                opt.step()

            self._agi.eval()
            with torch.no_grad():
                rs, rt = val if val is not None else (Xs, Xt)
                ys, yt = self._agi(rs, rt)
                dv = float(mse(ys, rs) + self.tel_agirlik * mse(yt, rt))
            self.gecmis_.append(dv)
            if dv < en_iyi - 1e-5:
                en_iyi, bekle = dv, 0
                en_iyi_durum = {k: v.detach().clone() for k, v in self._agi.state_dict().items()}
            else:
                bekle += 1
                if bekle >= self.sabir:
                    break
        if en_iyi_durum is not None:
            self._agi.load_state_dict(en_iyi_durum)
        return self

    def _ileri(self, X_spek: np.ndarray, X_tel: np.ndarray) -> tuple[torch.Tensor, ...]:
        if self._agi is None:
            raise RuntimeError(f"{self.ad}.egit() çağrılmadı")
        cihaz = cihaz_sec()
        self._agi.to(cihaz).eval()
        Xs = torch.tensor(_dogrula(X_spek, "X_spek", 3), device=cihaz)
        Xt = torch.tensor(_dogrula(X_tel, "X_tel", 2), device=cihaz)
        with torch.no_grad():
            ys, yt = self._agi(Xs, Xt)
        return Xs, Xt, ys, yt

    def anomali_skoru(self, X_spek: np.ndarray, X_tel: np.ndarray) -> np.ndarray:
        Xs, Xt, ys, yt = self._ileri(X_spek, X_tel)
        e_spek = ((ys - Xs) ** 2).mean(dim=(1, 2))
        e_tel = ((yt - Xt) ** 2).mean(dim=1)
        return (e_spek + self.tel_agirlik * e_tel).cpu().numpy()

    def anomali_skoru_dallar(self, X_spek: np.ndarray, X_tel: np.ndarray) -> dict[str, np.ndarray]:
        """Ablasyon/analiz için ayrı dal hataları: toplam, yalnız spektrogram, yalnız telemetri."""
        Xs, Xt, ys, yt = self._ileri(X_spek, X_tel)
        e_spek = ((ys - Xs) ** 2).mean(dim=(1, 2)).cpu().numpy()
        e_tel = ((yt - Xt) ** 2).mean(dim=1).cpu().numpy()
        return {"toplam": e_spek + self.tel_agirlik * e_tel, "spektrogram": e_spek, "telemetri": e_tel}
