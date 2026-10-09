"""Memeriksa setiap bilangan Contoh Soal Bab 6."""
import numpy as np

from bab01_data import WK, WQ, WV, data_mini, lintasan_maju, mask_kausal
from bab06_multihead import kv_cache, mha

r4 = lambda v: np.round(v, 4)
X = data_mini()
O, A, H = mha(X, WQ, WK, WV, np.eye(2), h=2, M=mask_kausal(3))

# Contoh Soal 6.1: dua cara menyusun keluaran
WO = np.array([[1.0, 1], [1, -1]])
g = np.array([-0.4621, 1.0])
assert np.allclose(g @ WO, [0.5379, -1.4621])
assert np.allclose(g[0] * WO[0] + g[1] * WO[1], g @ WO)
assert WO[0] @ WO[1] == 0
O2, _, H2 = mha(X, WQ, WK, WV, WO, h=2, M=mask_kausal(3))
assert np.allclose(O2, H2[0] @ WO[:1] + H2[1] @ WO[1:])
assert np.allclose(r4(O2[1]), [0.5379, -1.4621])

# kolom per head
Q, K, V = X @ WQ, X @ WK, X @ WV
assert np.array_equal(Q[:, 0], [1, 1, 2]) and np.array_equal(K[:, 0], [0, 1, 1])
assert np.array_equal(V[:, 0], [1, -1, 0])
assert np.array_equal(Q[:, 1], [0, 1, 1]) and np.array_equal(K[:, 1], [1, 0, 1])
assert np.array_equal(V[:, 1], [1, 1, 2])

# Contoh Soal 6.2: head 1
e = np.e
assert round(1 + e, 4) == 3.7183
assert np.allclose(r4(A[0, 1, :2]), [0.2689, 0.7311])
assert round(H[0, 1, 0], 4) == -0.4621
assert round(np.exp(2), 4) == 7.3891 and round(1 + 2 * np.exp(2), 4) == 15.7781
assert np.allclose(r4(A[0, 2]), [0.0634, 0.4683, 0.4683])
assert round(H[0, 2, 0], 4) == -0.4049
assert np.isclose(round(0.0634 - 0.4683, 4), -0.4049)
assert H[0, 0, 0] == 1

# Contoh Soal 6.3: head 2 dan sambungan
assert np.allclose(r4(A[1, 1, :2]), [0.7311, 0.2689])
assert np.isclose(H[1, 1, 0], 1)
assert round(1 + 2 * e, 4) == 6.4366
assert np.allclose(r4(A[1, 2]), [0.4223, 0.1554, 0.4223])
assert round(H[1, 2, 0], 4) == 1.4223
assert np.allclose(r4(O), [[1, 1], [-0.4621, 1], [-0.4049, 1.4223]])
O1 = lintasan_maju(X)["O"]
assert np.allclose(r4(O1), [[1, 1], [0, 1], [-0.1440, 1.5760]])
assert np.allclose(A.sum(-1), 1) and np.allclose(O[0], O1[0])

# teks: parameter dan perkalian tidak bergantung pada h
d, n = 768, 1024
for h in (1, 2, 12, 64):
    assert h * n * n * (d // h) == n * n * d

# Contoh Soal 6.4: KV cache
b, by = kv_cache(32, 32, 128)
assert b == 262_144 and by == 524_288 == 512 * 1024
assert 4096 * 512 / 1024 == 2048
b8, by8 = kv_cache(32, 8, 128)
assert b8 == 65_536 and by8 == 128 * 1024 and 4096 * 128 / 1024 == 512
b1, by1 = kv_cache(32, 1, 128)
assert b1 == 8_192 and by1 == 16 * 1024 and 4096 * 16 / 1024 == 64
print("Bab 6: semua Contoh Soal cocok")
