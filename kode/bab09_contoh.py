"""Memeriksa setiap bilangan Contoh Soal Bab 9."""
import numpy as np

from bab01_data import data_mini
from bab09_normalisasi import layernorm, layernorm_mundur, rmsnorm

r4 = lambda v: np.round(v, 4)

# Contoh Soal 9.1: LayerNorm (0, 2, 4)
x = np.array([[0.0, 2, 4]])
y, c = layernorm(x, eps=0)
assert np.isclose(c[1][0, 0] ** 2, 8 / 3) and round(c[1][0, 0], 4) == 1.6330
assert np.allclose(r4(y), [[-1.2247, 0, 1.2247]])
assert np.isclose(y[0, 2], np.sqrt(1.5))
assert np.isclose(y.sum(), 0) and np.isclose((y**2).sum(), 3)
y2, _ = layernorm(np.array([[1.0, 2, 6]]), eps=0)
assert np.allclose(r4(y2), [[-0.9258, -0.4629, 1.3887]])
assert np.isclose((y2**2).sum(), 3)

# Contoh Soal 9.2: gradien
dX, _, _ = layernorm_mundur(np.array([[1.0, 0, 0]]), c)
xh = y[0]
assert round(np.mean([1, 0, 0]), 4) == 0.3333
assert round(np.mean(np.array([1, 0, 0]) * xh), 4) == -0.4082
assert round(0.4082 * 1.2247, 4) == 0.4999 and np.isclose(
    np.sqrt(1 / 6) * np.sqrt(1.5), 0.5)
dalam = np.array([1, 0, 0]) - 1 / 3 - xh * np.mean(np.array([1, 0, 0]) * xh)
assert np.allclose(r4(dalam), [0.1667, -0.3333, 0.1667])
assert np.allclose(r4(dX), [[0.1021, -0.2041, 0.1021]])
assert np.allclose(dX, np.array([[1, -2, 1]]) / (6 * c[1][0, 0]))
assert round(6 * c[1][0, 0], 3) == 9.798
assert np.isclose(dX.sum(), 0) and np.isclose(dX @ xh, 0)
# Jacobian sebagai proyeksi
d = 3
P = np.eye(d) - np.ones((d, d)) / d - np.outer(xh, xh) / d
assert np.allclose(P @ P, P) and np.allclose(P @ np.ones(d), 0)
assert np.allclose(dX[0], P @ np.array([1, 0, 0]) / c[1][0, 0])

# Contoh Soal 9.3: RMSNorm
r = rmsnorm(x, eps=0)
assert round(np.sqrt(20 / 3), 4) == 2.5820
assert np.allclose(r4(r), [[0, 0.7746, 1.5492]])
assert np.isclose((r**2).sum(), 3) and round(0.7746**2, 1) == 0.6
assert round(1.5492**2, 1) == 2.4

# Contoh Soal 9.4: data mini
Y, _ = layernorm(data_mini(), eps=1e-5)
assert np.allclose(np.round(Y, 5), [[0.99998, -0.99998], [-0.99998, 0.99998],
                                    [0, 0]])
assert np.allclose(Y[2], 0)
print("Bab 9: semua Contoh Soal cocok")
