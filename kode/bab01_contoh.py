"""Memeriksa setiap bilangan Contoh Soal Bab 1."""
import numpy as np

from bab01_alur import banyak_parameter
from bab01_data import (E, MASUK, SASARAN, W1, W2, WK, WQ, WV, data_mini,
                        lintasan_maju, loss_ce, one_hot)

r4 = lambda v: np.round(v, 4)

# Contoh Soal 1.1: X = O E
O = one_hot(MASUK)
X = O @ E
assert O.shape == (3, 4) and X.shape == (3, 2)
assert np.array_equal(X, [[1, 0], [0, 1], [1, 1]])
assert np.array_equal(X, data_mini())

# Contoh Soal 1.2: Q, K, V, QK^T
Q, K, V = X @ WQ, X @ WK, X @ WV
assert np.array_equal(Q, [[1, 0], [1, 1], [2, 1]])
assert np.array_equal(K, [[0, 1], [1, 0], [1, 1]])
assert np.array_equal(V, [[1, 1], [-1, 1], [0, 2]])
assert np.array_equal(Q @ K.T, [[0, 1, 1], [1, 1, 2], [1, 2, 3]])

# Contoh Soal 1.3: A, O, H
h = lintasan_maju(X)
s3 = np.array([1, 2, 3]) / np.sqrt(2)
assert np.allclose(r4(s3), [0.7071, 1.4142, 2.1213])
e3 = np.exp(s3)
assert np.allclose(r4(e3), [2.0281, 4.1133, 8.3421])
assert round(e3.sum(), 4) == 14.4835
a3 = e3 / e3.sum()
assert np.allclose(r4(a3), [0.1400, 0.2840, 0.5760])
assert np.allclose(h["A"][:2], [[1, 0, 0], [0.5, 0.5, 0]])
assert np.allclose(h["O"][:2], [[1, 1], [0, 1]])
assert np.allclose(r4(h["O"][2]), [-0.1440, 1.5760])
assert np.allclose(r4(h["H"]), [[2, 1], [0, 2], [0.8560, 2.5760]])

# Contoh Soal 1.4: FFN, logit, loss posisi 1 dan 2
assert np.array_equal(h["H"][0] @ W1, [2, 1, 3, -1])
assert np.array_equal(h["Z"][0], [2, 1, 3, 0])
assert np.array_equal(h["H"][1] @ W1, [0, 2, 2, 2])
assert np.allclose(h["F"][:2], [[-0.5, 0.5], [-1, 0]])
assert np.allclose(h["H2"][:2], [[1.5, 1.5], [-1, 2]])
assert np.allclose(h["logit"][:2], [[1.5, 1.5, 3, 0], [-1, 2, 1, 3]])
j1 = np.exp(h["logit"][0]).sum()
j2 = np.exp(h["logit"][1]).sum()
assert np.allclose(r4(np.exp(h["logit"][0])), [4.4817, 4.4817, 20.0855, 1])
assert round(j1, 4) == 30.0489 and round(np.log(j1), 4) == 3.4028
assert round(np.log(j1) - 1.5, 4) == 1.9028
assert np.allclose(r4(np.exp(h["logit"][1])),
                   [0.3679, 7.3891, 2.7183, 20.0855])
assert round(j2, 4) == 30.5608 and round(np.log(j2) - 1, 4) == 2.4197
assert round(4.4817 / 30.0489, 4) == 0.1491 == round(np.exp(-1.90285), 4)
P = h["P"]
assert round(-np.log(P[0, SASARAN[0]]), 4) == 1.9028
assert round(-np.log(P[1, SASARAN[1]]), 4) == 2.4197

# teks: loss rata-rata, peluang kalimat, seragam
L = loss_ce(P, SASARAN)
assert round(L, 4) == 1.6868
pk = np.prod(P[np.arange(3), SASARAN])
assert round(np.exp(-3 * L), 4) == round(pk, 4) == 0.0063
assert np.allclose(r4(P[np.arange(3), SASARAN]), [0.1491, 0.0889, 0.4781])
assert round(np.log(4), 4) == 1.3863 and L > np.log(4)
# turunan logit = (P - one-hot)/n, diperiksa dengan beda hingga
Z = h["logit"].copy()
eps = 1e-6
g = (P - one_hot(SASARAN)) / 3


def _L(Z):
    Zs = Z - Z.max(1, keepdims=True)
    return np.mean(np.log(np.exp(Zs).sum(1)) - Zs[np.arange(3), SASARAN])


for t in range(3):
    for v in range(4):
        Zp = Z.copy()
        Zp[t, v] += eps
        assert abs((_L(Zp) - _L(Z)) / eps - g[t, v]) < 1e-5

# Contoh Soal 1.5: parameter GPT-2 kecil
assert 12 * 768**2 + 13 * 768 == 7_087_872 and 12 * 768**2 == 7_077_888
assert 13 * 768 == 9_984 and 12 * 7_087_872 == 85_054_464
assert 50257 * 768 == 38_597_376 and 1024 * 768 == 786_432
tot = 85_054_464 + 38_597_376 + 786_432 + 1_536
assert tot == 124_439_808
r = banyak_parameter(768, 12, 3072, 50257, 1024)
assert r["total"] == tot and r["lapisan"] == 7_087_872
assert r["emb"] == 39_383_808 and round(100 * r["emb"] / tot, 1) == 31.6
# rumus umum 12 d^2 + 13 d
for d in (2, 64, 768):
    assert banyak_parameter(d, 1, 4 * d, 1, 0, ln_akhir=False)[
        "lapisan"] == 12 * d * d + 13 * d
print("Bab 1: semua Contoh Soal cocok")
