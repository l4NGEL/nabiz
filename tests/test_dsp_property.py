"""Test katmanı 1 — property-based (hypothesis): rastgele geçerli sinyal → değişmezler.

Aranan: çökme yok, şekil değişmezleri, sonluluk, determinizm, sınırlılık.
"""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nabiz import config
from nabiz.dsp import features as F
from nabiz.dsp import hizalama as H
from nabiz.dsp import telemetri as TL

SR = config.ORNEKLEME_HIZI

# 0.3–1.0 sn arası, [-1, 1] aralığında sonlu sinyaller
_sinyal = st.integers(min_value=int(0.3 * SR), max_value=SR).flatmap(
    lambda n: arrays(
        np.float64, n,
        elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=32),
    )
)


@given(x=_sinyal)
@settings(max_examples=60, deadline=None)
def test_log_mel_degismezleri(x):
    lm = F.log_mel_spektrogram(x)
    assert lm.shape[0] == config.N_MEL
    assert lm.ndim == 2 and lm.shape[1] >= 1
    assert np.all(np.isfinite(lm))
    np.testing.assert_array_equal(lm, F.log_mel_spektrogram(x))  # determinizm


@given(x=_sinyal)
@settings(max_examples=60, deadline=None)
def test_pencere_oznitelikleri_sonlu_ve_sabit_anahtar(x):
    oz = F.pencere_oznitelikleri(x)
    assert len(oz) == 2 * config.N_MEL + 2 + 6  # a_mel_ort/std + rms + kurtosis + z_*
    assert all(np.isfinite(v) for v in oz.values())


@given(
    x=_sinyal,
    kaydir=st.integers(min_value=-200, max_value=200),
)
@settings(max_examples=40, deadline=None)
def test_capraz_korelasyon_kayma_sinirli(x, kaydir):
    if x.size < 500:
        return
    y = np.roll(x, kaydir)
    kayma = H.capraz_korelasyon_kaymasi(x, y, SR, maks_kayma_sn=0.05)
    assert abs(kayma) <= 0.05 + 1e-9


@given(
    n=st.integers(min_value=2, max_value=200),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=40, deadline=None)
def test_telemetri_oznitelik_her_zaman_sonlu(n, seed):
    rng = np.random.default_rng(seed)
    seri = {
        "rpm": rng.normal(1800, 50, n),
        "yuk": rng.uniform(0, 1, n),
        "sicaklik": rng.normal(50, 5, n),
        "gerilim": rng.normal(400, 3, n),
    }
    oz = TL.telemetri_oznitelik(seri)
    assert set(oz) == set(TL.telemetri_oznitelik_adlari())
    assert all(np.isfinite(v) for v in oz.values())


@given(
    hiz_a=st.sampled_from([8000, 16000, 22050, 25600]),
    hiz_b=st.sampled_from([8000, 16000, 25600, 48000]),
)
@settings(max_examples=25, deadline=None)
def test_hizala_ortak_uzunluk_degismezi(hiz_a, hiz_b):
    t_a = np.arange(int(1.5 * hiz_a)) / hiz_a
    t_b = np.arange(int(1.5 * hiz_b)) / hiz_b
    kanallar = {
        "a": H.Kanal(np.sin(2 * np.pi * 100 * t_a), hiz_a, ad="a"),
        "b": H.Kanal(np.sin(2 * np.pi * 100 * t_b), hiz_b, ad="b"),
    }
    hedef = max(hiz_a, hiz_b)
    s = H.hizala(kanallar, hedef_hiz=hedef)
    boylar = {v.size for v in s.kanallar.values()}
    assert len(boylar) == 1
    assert s.hizalama_hatasi_sn < 1e-3
