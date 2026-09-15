"""RUL (kalan faydalı ömür) regresyon modeli — C-MAPSS.

Girdi: (pencere, n_sensor) çok-değişkenli zaman penceresi. GRU son gizli
durumu -> küçük MLP kafa -> skaler RUL tahmini. MSE ile eğitilir;
değerlendirme MAE/RMSE + C-MAPSS asimetrik skoru (eval/metrics.py::cmapss_skoru
— geç tahmin, RUL'u olduğundan yüksek göstermek, daha ağır cezalı).

Autoencoder'lardan farklı arayüz (X, y çifti) — `models._egitim.TemelAE`'yi
genişletmez; aynı desenler (ağ-öncesi tohumlama, erken durma, küçük-veri
batch boyutu) burada tekrarlanır.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nabiz.models._egitim import cihaz_sec


def _dogrula(X: np.ndarray, ad: str, ndim: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != ndim:
        raise ValueError(f"{ad} {ndim}B olmalı, shape={X.shape}")
    if X.size == 0:
        raise ValueError(f"{ad} boş")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{ad} sonlu değil (NaN/inf)")
    return X


class _RULAgi(nn.Module):
    def __init__(self, n_sensor: int, gizli: int, gru_kat: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_sensor, gizli, gru_kat, batch_first=True)
        self.kafa = nn.Sequential(nn.Linear(gizli, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, pencere, n_sensor)
        _, son = self.gru(x)
        return self.kafa(son[-1]).squeeze(-1)  # (N,)


class RULRegresor:
    ad = "rul_gru"

    def __init__(
        self,
        gizli: int = 64,
        gru_kat: int = 1,
        epok: int = 150,
        yigin: int = 64,
        ogrenme_hizi: float = 1e-3,
        sabir: int = 15,
        seed: int = 0,
    ) -> None:
        self.gizli, self.gru_kat = gizli, gru_kat
        self.epok, self.yigin = epok, yigin
        self.ogrenme_hizi, self.sabir, self.seed = ogrenme_hizi, sabir, seed
        self._agi: _RULAgi | None = None
        self.gecmis_: list[float] = []
        # hedef (RUL, çevrim) standardizasyonu — ham ölçekte (0-125) optimizasyon
        # yavaş yakınsıyor/ortalamaya sıkışıyor; dışa açık arayüz yine ham RUL.
        self.y_ort_: float | None = None
        self.y_std_: float | None = None

    def egit(
        self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None
    ) -> RULRegresor:
        X, y = _dogrula(X, "X", 3), _dogrula(y, "y", 1)
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) ve y ({len(y)}) örnek sayısı uyuşmuyor")

        self.y_ort_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        y_norm = (y - self.y_ort_) / self.y_std_

        cihaz = cihaz_sec()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._agi = _RULAgi(X.shape[2], self.gizli, self.gru_kat).to(cihaz)
        opt = torch.optim.Adam(self._agi.parameters(), lr=self.ogrenme_hizi)
        mse = nn.MSELoss()

        Xt, yt = torch.tensor(X, device=cihaz), torch.tensor(y_norm, device=cihaz)
        val = None
        if X_val is not None and y_val is not None and len(X_val):
            y_val_norm = (_dogrula(y_val, "y_val", 1) - self.y_ort_) / self.y_std_
            val = (
                torch.tensor(_dogrula(X_val, "X_val", 3), device=cihaz),
                torch.tensor(y_val_norm, device=cihaz),
            )

        etkin = max(8, min(self.yigin, len(Xt) // 8 or len(Xt)))
        yukleyici = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=etkin, shuffle=True, drop_last=len(Xt) >= 2 * etkin,
        )

        en_iyi, en_iyi_durum, bekle = float("inf"), None, 0
        for _ep in range(self.epok):
            self._agi.train()
            for bx, by in yukleyici:
                opt.zero_grad()
                mse(self._agi(bx), by).backward()
                opt.step()
            self._agi.eval()
            with torch.no_grad():
                rx, ry = val if val is not None else (Xt, yt)
                dv = float(mse(self._agi(rx), ry))
            self.gecmis_.append(dv)
            if dv < en_iyi - 1e-4:
                en_iyi, bekle = dv, 0
                en_iyi_durum = {k: v.detach().clone() for k, v in self._agi.state_dict().items()}
            else:
                bekle += 1
                if bekle >= self.sabir:
                    break
        if en_iyi_durum is not None:
            self._agi.load_state_dict(en_iyi_durum)
        return self

    def tahmin(self, X: np.ndarray) -> np.ndarray:
        if self._agi is None:
            raise RuntimeError(f"{self.ad}.egit() çağrılmadı")
        cihaz = cihaz_sec()
        self._agi.to(cihaz).eval()
        with torch.no_grad():
            Xt = torch.tensor(_dogrula(X, "X", 3), device=cihaz)
            norm = self._agi(Xt).cpu().numpy()
        return norm * self.y_std_ + self.y_ort_
