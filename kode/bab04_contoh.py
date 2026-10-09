"""Memeriksa setiap bilangan Contoh Soal Bab 4."""
import itertools

import numpy as np

from bab01_data import WK, WQ, WV, data_mini, mask_kausal, softmax
from bab04_attention import attn, entropi, nadaraya_watson

r4 = lambda v: np.round(v, 4)
X = data_mini()
Q, K, V = X @ WQ, X @ WK, X @ WV

# Contoh Soal 4.1: bentuk bilinear
W = WQ @ WK.T
assert np.array_equal(WK, WK.T) and np.array_equal(W, [[0, 1], [1, 1]])
assert np.array_equal(W @ X[2], [1, 2]) and X[2] @ W @ X[2] == 3
assert X[0] @ W @ X[1] == 1 == X[1] @ W @ X[0]
assert np.allclose(X @ W @ X.T, Q @ K.T)
W2 = np.eye(2) @ np.array([[1, 1], [0, 1]]).T
assert np.array_equal(W2, [[1, 0], [1, 1]])
assert X[0] @ W2 @ X[1] == 0 and X[1] @ W2 @ X[0] == 1
# invarian terhadap B
B = np.array([[2.0, 1], [0, 3]])
assert np.allclose((WQ @ B) @ (WK @ np.linalg.inv(B).T).T, W)

# Contoh Soal 4.2: q.k untuk tanda acak, d_k = 4
nilai = [sum(a * b for a, b in zip(q, k))
         for q in itertools.product([-1, 1], repeat=4)
         for k in itertools.product([-1, 1], repeat=4)]
assert np.isclose(np.mean(nilai), 0) and np.isclose(np.var(nilai), 4)
assert max(nilai) == 4 and np.isclose(np.var(np.array(nilai) / 2), 1)
assert 4 * 4 * 0.5 * 0.5 == 4

# teks: rasio e^16
assert round(np.exp(16) / 1e6) == 9

# Contoh Soal 4.3: skala
assert np.allclose(softmax(0 * np.array([1, 2, 3])), 1 / 3)
e = np.exp([1, 2, 3])
assert np.allclose(r4(e), [2.7183, 7.3891, 20.0855])
assert round(e.sum(), 4) == 30.1929
a1 = softmax([1, 2, 3])
assert np.allclose(r4(a1), [0.0900, 0.2447, 0.6652])
assert np.allclose(softmax(50 * np.array([1, 2, 3])), [0, 0, 1], atol=1e-20)
assert np.allclose(softmax([1000, 1001, 1002]), a1)
assert round(1 / np.sqrt(2), 4) == 0.7071
assert round(np.finfo(float).max / 1e308, 1) == 1.8
assert 1000 > np.log(np.finfo(float).max)

# Contoh Soal 4.4: encoder tanpa mask
O, A = attn(Q, K, V)
e1 = np.exp(np.array([0, 1, 1]) / np.sqrt(2))
assert np.allclose(r4(e1), [1, 2.0281, 2.0281]) and round(e1.sum(), 4) == 5.0562
assert np.allclose(r4(A[0]), [0.1978, 0.4011, 0.4011])
e2 = np.exp(np.array([1, 1, 2]) / np.sqrt(2))
assert np.allclose(r4(e2), [2.0281, 2.0281, 4.1133])
assert round(e2.sum(), 4) == 8.1695
assert np.allclose(r4(A[1]), [0.2483, 0.2483, 0.5035])
assert np.allclose(O[:, 0], A[:, 0] - A[:, 1])
assert np.allclose(O[:, 1], 1 + A[:, 2])
assert np.allclose(r4(O[:2]), [[-0.2033, 1.4011], [0, 1.5035]])
Ok, Ak = attn(Q, K, V, M=mask_kausal(3))
assert np.allclose(A[2], Ak[2]) and np.allclose(r4(A[2]), [.14, .284, .576])
from bab03_posisi import attention
_, Op = attention(X[[1, 2, 0]], kausal=True)
assert np.allclose(Op[2], O[0])

# Contoh Soal 4.5: entropi
H = entropi(Ak)
assert np.isclose(H[0], 0) and round(H[1], 4) == 0.6931
suku = -Ak[2] * np.log(Ak[2])
assert np.allclose(np.round(suku, 5), [0.27528, 0.35749, 0.31776])
assert round(0.27528 + 0.35749 + 0.31776, 4) == 0.9505
assert round(H[2], 4) == 0.9505 and round(np.log(3), 4) == 1.0986
assert np.all(entropi(A) <= np.log(3)) and np.all(entropi(A)[:2] > 0)

# Contoh Soal 4.6: Nadaraya-Watson
xs, ys = np.array([0.0, 1, 2]), np.array([1.0, 3, 2])
w = np.exp(-np.array([1, 0, 1]) / 2)
assert np.allclose(r4(w), [0.6065, 1, 0.6065]) and round(w.sum(), 4) == 2.2131
assert np.allclose(r4(w / w.sum()), [0.2741, 0.4519, 0.2741])
f = nadaraya_watson(np.array([1.0]), xs, ys, 1.0)[0]
assert round(f, 4) == 2.1778
skor = 1 * xs - xs**2 / 2
assert np.allclose(skor, [0, 0.5, 0])
assert np.allclose(softmax(skor), w / w.sum())
print("Bab 4: semua Contoh Soal cocok")
