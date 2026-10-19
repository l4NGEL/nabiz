"""Tip başına izleme skorları: Ledoit-Wolf uzaklığı, PCA artığı (SPE), Hotelling T².

Düz Mahalanobis 130 boyutta kovaryansı şişirir. SPE, öğrenilen altuzayın
dışına taşan arızayı yakalar; T² altuzayın içindeki kaymayı. Her tip kendi
normaliyle eğitilir.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from nabiz.models.temel import dogrula_2b


class LedoitUzaklik:
    """Çekilmiş kovaryans; skor = Mahalanobis uzaklığı."""

    ad = "ledoit"

    def __init__(self) -> None:
        self._m: LedoitWolf | None = None

    def egit(self, X: np.ndarray) -> LedoitUzaklik:
        self._m = LedoitWolf().fit(dogrula_2b(X))
        return self

    def anomali_skoru(self, X: np.ndarray) -> np.ndarray:
        if self._m is None:
            raise RuntimeError("LedoitUzaklik.egit() çağrılmadı")
        return np.sqrt(np.clip(self._m.mahalanobis(dogrula_2b(X)), 0.0, None))


class _TipTaban:
    def __init__(self) -> None:
        self._modeller: dict = {}
        self._yedek = None

    def _yedir(self, X: np.ndarray, tipler: np.ndarray, kur) -> None:
        X = dogrula_2b(X)
        tipler = np.asarray(tipler).ravel()
        if tipler.size != X.shape[0]:
            raise ValueError("tipler ve X aynı uzunlukta olmalı")
        self._modeller = {}
        for tip in np.unique(tipler):
            m = tipler == tip
            if int(m.sum()) < 3:
                continue
            self._modeller[str(tip)] = kur(X[m])
        self._yedek = kur(X)

    def _skorla(self, X: np.ndarray, tipler: np.ndarray, fn) -> np.ndarray:
        if self._yedek is None:
            raise RuntimeError("egit() çağrılmadı")
        X = dogrula_2b(X)
        tipler = np.asarray(tipler).ravel()
        skor = np.empty(X.shape[0], dtype=np.float64)
        for tip in np.unique(tipler):
            m = tipler == tip
            model = self._modeller.get(str(tip), self._yedek)
            skor[m] = fn(model, X[m])
        return skor


def _pca_kur(n_bilesen: int):
    def kur(X: np.ndarray) -> PCA:
        n = min(n_bilesen, X.shape[0] - 1, X.shape[1])
        return PCA(n_components=max(1, n), random_state=0).fit(X)

    return kur


class TipSpe(_TipTaban):
    """PCA artığı — altuzayın açıklayamadığı enerji. Yüksek = arıza."""

    def __init__(self, n_bilesen: int = 32) -> None:
        super().__init__()
        self.ad = f"spe{n_bilesen}"
        self._n = n_bilesen

    def egit(self, X: np.ndarray, tipler: np.ndarray) -> TipSpe:
        self._yedir(X, tipler, _pca_kur(self._n))
        return self

    def anomali_skoru(self, X: np.ndarray, tipler: np.ndarray) -> np.ndarray:
        def fn(pca: PCA, blok: np.ndarray) -> np.ndarray:
            z = pca.transform(blok)
            geri = pca.inverse_transform(z)
            return np.sum((blok - geri) ** 2, axis=1)

        return self._skorla(X, tipler, fn)


class TipT2(_TipTaban):
    """Hotelling T² — altuzay içindeki uzaklık."""

    def __init__(self, n_bilesen: int = 32) -> None:
        super().__init__()
        self.ad = f"t2_{n_bilesen}"
        self._n = n_bilesen

    def egit(self, X: np.ndarray, tipler: np.ndarray) -> TipT2:
        self._yedir(X, tipler, _pca_kur(self._n))
        return self

    def anomali_skoru(self, X: np.ndarray, tipler: np.ndarray) -> np.ndarray:
        def fn(pca: PCA, blok: np.ndarray) -> np.ndarray:
            z = pca.transform(blok)
            lam = np.maximum(pca.explained_variance_, 1e-12)
            return np.sum((z ** 2) / lam, axis=1)

        return self._skorla(X, tipler, fn)


class TipGmm(_TipTaban):
    """PCA altuzayında karışım modeli. Skor = −log olabilirlik."""

    def __init__(self, n_bilesen: int = 16, n_karisik: int = 2) -> None:
        super().__init__()
        self.ad = f"gmm{n_bilesen}x{n_karisik}"
        self._n = n_bilesen
        self._k = n_karisik

    def egit(self, X: np.ndarray, tipler: np.ndarray) -> TipGmm:
        def kur(blok: np.ndarray):
            n = max(1, min(self._n, blok.shape[0] - 1, blok.shape[1]))
            pca = PCA(n_components=n, random_state=0).fit(blok)
            k = min(self._k, max(1, blok.shape[0] // 20), n)
            gmm = GaussianMixture(
                n_components=max(1, k), covariance_type="full",
                reg_covar=1e-3, random_state=0, max_iter=80,
            )
            gmm.fit(pca.transform(blok))
            return pca, gmm

        self._yedir(X, tipler, kur)
        return self

    def anomali_skoru(self, X: np.ndarray, tipler: np.ndarray) -> np.ndarray:
        def fn(model, blok: np.ndarray) -> np.ndarray:
            pca, gmm = model
            return -gmm.score_samples(pca.transform(blok))

        return self._skorla(X, tipler, fn)


class TipKnn(_TipTaban):
    """PCA altuzayında en yakın normal komşuların uzaklığı."""

    def __init__(self, n_bilesen: int = 32, komsu: int = 5, tavan: int = 8000) -> None:
        super().__init__()
        self.ad = f"knn{n_bilesen}"
        self._n = n_bilesen
        self._k = komsu
        self._tavan = tavan

    def egit(self, X: np.ndarray, tipler: np.ndarray) -> TipKnn:
        def kur(blok: np.ndarray):
            n = max(1, min(self._n, blok.shape[0] - 1, blok.shape[1]))
            pca = PCA(n_components=n, random_state=0).fit(blok)
            Z = pca.transform(blok)
            if Z.shape[0] > self._tavan:
                sec = np.random.default_rng(0).choice(Z.shape[0], self._tavan, replace=False)
                Z = Z[sec]
            k = min(self._k, Z.shape[0] - 1)
            nn = NearestNeighbors(n_neighbors=max(1, k), algorithm="brute").fit(Z)
            return pca, nn

        self._yedir(X, tipler, kur)
        return self

    def anomali_skoru(self, X: np.ndarray, tipler: np.ndarray) -> np.ndarray:
        def fn(model, blok: np.ndarray) -> np.ndarray:
            pca, nn = model
            uzaklik, _ = nn.kneighbors(pca.transform(blok))
            return uzaklik.mean(axis=1)

        return self._skorla(X, tipler, fn)


class TipKuyruk(_TipTaban):
    """İlk bileşenler atılır; kalanların standart T²'si (kuyruk altuzayı)."""

    def __init__(self, koru: int = 64) -> None:
        super().__init__()
        self.ad = f"kuyruk{koru}"
        self._koru = koru

    def egit(self, X: np.ndarray, tipler: np.ndarray) -> TipKuyruk:
        def kur(blok: np.ndarray) -> PCA:
            n = min(blok.shape[0] - 1, blok.shape[1])
            return PCA(n_components=max(2, n), random_state=0).fit(blok)

        self._yedir(X, tipler, kur)
        return self

    def anomali_skoru(self, X: np.ndarray, tipler: np.ndarray) -> np.ndarray:
        def fn(pca: PCA, blok: np.ndarray) -> np.ndarray:
            z = pca.transform(blok)
            lam = np.maximum(pca.explained_variance_, 1e-12)
            k = min(self._koru, z.shape[1] - 1)
            return np.sum((z[:, k:] ** 2) / lam[k:], axis=1)

        return self._skorla(X, tipler, fn)
