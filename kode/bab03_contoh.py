"""Memeriksa setiap bilangan Contoh Soal Bab 3."""
import numpy as np

from bab01_data import WK, WQ, WV, data_mini, softmax
from bab03_posisi import attention, bias_alibi, pe_sinus, rope

r4 = lambda v: np.round(v, 4)
X = data_mini()

# Contoh Soal 3.1: permutasi ilmu itu <awal>
perm = [1, 2, 0]
_, Op = attention(X[perm], kausal=True)
assert np.allclose(Op[0], [-1, 1]) and np.allclose(X[1] @ WV, [-1, 1])
_, O = attention(X, kausal=True)
assert np.allclose(O[1], [0, 1])
_, O0 = attention(X, kausal=False)
_, O0p = attention(X[perm], kausal=False)
assert np.allclose(O0p, O0[perm])
# bukti: P S P^T dan P A P^T
P = np.eye(3)[perm]
S = (X @ WQ) @ (X @ WK).T
Sp = (P @ X @ WQ) @ (P @ X @ WK).T
assert np.allclose(Sp, P @ S @ P.T)
assert np.allclose(softmax(Sp), P @ softmax(S) @ P.T)

# Contoh Soal 3.2: sinusoidal d = 2
PE = pe_sinus(3, 2)
assert np.allclose(r4(PE), [[0, 1], [0.8415, 0.5403], [0.9093, -0.4161]])
assert np.allclose(r4(X + PE), [[1, 1], [0.8415, 1.5403],
                                [1.9093, 0.5839]])
assert round(PE[0] @ PE[2], 4) == -0.4161 == round(np.cos(2), 4)
assert np.allclose(np.linalg.norm(pe_sinus(10, 8), axis=1), 2)
# Persamaan rotasi untuk pergeseran k
w, p, k = 0.3, 5, 3
R = np.array([[np.cos(w * k), np.sin(w * k)],
              [-np.sin(w * k), np.cos(w * k)]])
assert np.allclose(R @ [np.sin(w * p), np.cos(w * p)],
                   [np.sin(w * (p + k)), np.cos(w * (p + k))])

# Contoh Soal 3.3: RoPE theta = pi/2
sud = np.arange(3)[:, None] * (np.pi / 2)
Qt, Kt = rope(X @ WQ, sud), rope(X @ WK, sud)
assert np.allclose(Qt, [[1, 0], [-1, 1], [-2, -1]])
assert np.allclose(Kt, [[0, 1], [0, 1], [-1, -1]])
St = Qt @ Kt.T
assert np.isclose(St[0, 0], 0) and np.allclose(St[1, :2], [1, 1])
assert np.allclose(St[2], [-1, -1, 3])
e = np.exp(np.array([-1, -1, 3]) / np.sqrt(2))
assert np.allclose(r4(e), [0.4931, 0.4931, 8.3421])
assert round(e.sum(), 4) == 9.3283
A, _ = attention(X, sudut=sud)
assert np.allclose(r4(A[2]), [0.0529, 0.0529, 0.8943])
Rm = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
q3, k1, q2 = X[2] @ WQ, X[0] @ WK, X[1] @ WQ
assert np.isclose(q3 @ Rm(-np.pi) @ k1, -1)
assert np.allclose(Rm(-np.pi / 2) @ k1, [1, 0]) and np.isclose(
    q2 @ Rm(-np.pi / 2) @ k1, 1)

# Contoh Soal 3.4: ALiBi m = 1/2
A, _ = attention(X, bias=bias_alibi(3, 0.5))
s2 = np.array([1, 1]) / np.sqrt(2) - [0.5, 0]
assert np.allclose(r4(s2), [0.2071, 0.7071])
assert round(1 + np.exp(0.5), 4) == 2.6487
assert np.allclose(r4(A[1, :2]), [0.3775, 0.6225])
s3 = np.array([1, 2, 3]) / np.sqrt(2) - [1, 0.5, 0]
assert np.allclose(r4(s3), [-0.2929, 0.9142, 2.1213])
assert np.allclose(r4(np.exp(s3)), [0.7461, 2.4948, 8.3421])
assert round(np.exp(s3).sum(), 4) == 11.5831
assert np.allclose(r4(A[2]), [0.0644, 0.2154, 0.7202])
assert np.allclose(A[0], [1, 0, 0])
print("Bab 3: semua Contoh Soal cocok")
