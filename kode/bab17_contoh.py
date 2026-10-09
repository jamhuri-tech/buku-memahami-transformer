"""Memeriksa setiap bilangan Contoh Soal Bab 17."""
import math

import numpy as np

from bab01_data import WK, WQ, WV, data_mini, softmax
from bab17_skala import attn_linear, chinchilla, phi

# Contoh Soal 17.1: memori A
n = 32768
assert n * n == 1_073_741_824 == 2**30 and 2 * n * n == 2**31
assert 2 * 32 * 32 == 2048
assert 2 * 32 * 32 * 128 * n * 2 == 16 * 2**30

# Contoh Soal 17.2: softmax daring
m, l = 2.0, math.exp(-1) + 1
assert round(l, 4) == 1.3679
l = l * math.exp(2 - 3) + 1
assert round(l - 1, 4) == 0.5032 and round(l, 4) == 1.5032
w = [math.exp(1 - 3) / l, math.exp(2 - 3) / l, 1 / l]
assert np.allclose(np.round(w, 4), [0.0900, 0.2447, 0.6652])
assert np.allclose(w, softmax([1, 2, 3]))
assert round(math.exp(-2), 4) == 0.1353

# Contoh Soal 17.3: attention linear
X = data_mini()
Q, K, V = X @ WQ, X @ WK, X @ WV
assert np.array_equal(phi(Q), Q + 1) and np.array_equal(phi(K), K + 1)
fq, fk = Q + 1, K + 1
assert list(fq[2] @ fk.T) == [7, 8, 10] and list(fq[1] @ fk[:2].T) == [6, 6]
O = attn_linear(Q, K, V)
assert np.allclose(O[1], [0, 1]) and np.allclose(O[2], [-0.04, 1.4])
assert np.allclose(np.array([7, 8, 10]) / 25, [0.28, 0.32, 0.40])

# Contoh Soal 17.4: MoE
p = 3 * 4096 * 14336
assert p == 176_160_768 and 8 * p == 1_409_286_144 and 2 * p == 352_321_536
assert 4096 * 8 == 32_768

# Contoh Soal 17.5: Chinchilla dan Gopher
assert np.isclose(6 * 7e10 * 1.4e12, 5.88e23)
assert np.isclose(6 * 2.8e11 * 3e11, 5.04e23)
a = 0.34 * math.log(7e10)
b = 0.28 * math.log(1.4e12)
assert round(a, 4) == 8.4904 and round(math.exp(a)) == 4868
assert round(406.4 / 4868, 4) == 0.0835
assert round(b, 4) == 7.8309 and round(math.exp(b)) == 2517
assert round(410.7 / 2517, 4) == 0.1632
assert round(1.69 + 0.0835 + 0.1632, 4) == 1.9367
assert round(chinchilla(7e10, 1.4e12), 4) == 1.9366
assert round(math.exp(0.34 * math.log(2.8e11))) == 7799
assert round(math.exp(0.28 * math.log(3e11))) == 1635
assert round(406.4 / 7799, 4) == 0.0521 and round(410.7 / 1635, 4) == 0.2512
assert round(chinchilla(2.8e11, 3e11), 4) == 1.9933
print("Bab 17: semua Contoh Soal cocok")
