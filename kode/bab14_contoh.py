"""Memeriksa setiap bilangan Contoh Soal Bab 14."""
import numpy as np

from bab01_data import data_mini, lintasan_maju
from bab14_decoding import beam, saring

r4 = lambda v: np.round(v, 4)

# Contoh Soal 14.1: beam search pada pohon kecil
P1 = {"": [0.6, 0.4]}
P2 = {"A": [0.4, 0.3, 0.3], "B": [0.9, 0.05, 0.05]}
def lp(b):
    if len(b) == 0:
        return np.log(np.array([0.6, 0.4, 1e-12]))
    return np.log(np.array(P2["AB"[b[0]]]))
g = beam(lp, [], 1, 2)[0]
bb = beam(lp, [], 2, 2)[0]
assert g[1] == [0, 0] and np.isclose(np.exp(g[0]), 0.24)
assert bb[1] == [1, 0] and np.isclose(np.exp(bb[0]), 0.36)
tot = [0.6 * x for x in P2["A"]] + [0.4 * x for x in P2["B"]]
assert np.allclose(tot, [0.24, 0.18, 0.18, 0.36, 0.02, 0.02])
assert np.isclose(sum(tot), 1)

# Contoh Soal 14.2: temperature, top-k, top-p
z = lintasan_maju(data_mini())["logit"][2]
assert np.allclose(np.round(z, 3), [-0.432, 3.004, 2.572, 3.436])
P = saring(z)
assert np.allclose(r4(P), [0.0100, 0.3104, 0.2015, 0.4781])
r = 2 * z - 2 * z.max()
assert round(2 * z.max(), 3) == 6.872
assert np.allclose(np.round(r, 3), [-7.736, -0.864, -1.728, 0])
e = np.exp(r)
assert np.allclose(r4(e), [0.0004, 0.4215, 0.1777, 1]) and round(e.sum(), 4) == 1.5996
assert np.allclose(r4(saring(z, suhu=0.5)), [0.0003, 0.2635, 0.1111, 0.6251])
assert np.allclose(r4(saring(z, k=2)), [0, 0.3937, 0, 0.6063])
assert np.allclose(saring(z, p=0.7), saring(z, k=2))
assert round(0.4781 + 0.3104, 4) == 0.7885
# P_T proporsional P^(1/T)
assert np.allclose(saring(z, suhu=0.5), P**2 / (P**2).sum())

# Contoh Soal 14.3: KV cache
assert 2 * 4 * 128 == 1024 and 1024 * 4 == 4096
assert 128 * 1024 == 131_072 and 131_072 * 4 == 512 * 1024
t = np.arange(1, 1025)
assert t.sum() == 524_800 and (t * (t + 1) // 2).sum() == 179_481_600
assert round(179_481_600 / 524_800) == 342
print("Bab 14: semua Contoh Soal cocok")
