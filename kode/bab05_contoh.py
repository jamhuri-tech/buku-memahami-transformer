"""Memeriksa setiap bilangan Contoh Soal Bab 5."""
import warnings

import numpy as np

from bab01_data import E, WK, WQ, WV, data_mini, mask_kausal, softmax
from bab04_attention import attn
from bab05_mask import mask_dari_boolean, mask_padding, pola_mask

r4 = lambda v: np.round(v, 4)
X = data_mini()
Q, K, V = X @ WQ, X @ WK, X @ WV

# Contoh Soal 5.1: awalan <awal> ilmu
Oa, Aa = attn(Q[:2], K[:2], V[:2])
assert Q[1] @ K[0] == 1 == Q[1] @ K[1]
assert np.allclose(Aa[1], [0.5, 0.5]) and np.allclose(Oa[1], [0, 1])
Ok, _ = attn(Q, K, V, M=mask_kausal(3))
assert np.allclose(Ok[1], Oa[1])
Ot, At = attn(Q, K, V)
assert np.allclose(r4(Ot[1]), [0, 1.5035])
assert np.isclose(Ot[1, 1] - 1, At[1, 2])

# Contoh Soal 5.2: <awal> itu <pad>
qi, k0, ki = E[2] @ WQ, E[0] @ WK, E[2] @ WK
assert np.array_equal(qi, [2, 1]) and qi @ k0 == 1 and qi @ ki == 3
s = np.array([1, 3]) / np.sqrt(2)
assert np.allclose(r4(s), [0.7071, 2.1213]) and round(s[1] - s[0], 4) == 1.4142
assert round(1 + np.exp(np.sqrt(2)), 4) == 5.1133
a = softmax(s)
assert np.allclose(r4(a), [0.1956, 0.8044])
o = a @ np.array([E[0] @ WV, E[2] @ WV])
assert np.allclose(r4(o), [0.1956, 1.8044])
Xb = np.stack([E[[0, 1, 2]], np.r_[E[[0, 2]], [[9.0, 9.0]]]])
M = mask_kausal(3)[None] + mask_padding([3, 2], 3)
Ob, Ab = attn(Xb @ WQ, Xb @ WK, Xb @ WV, M=M)
assert np.allclose(Ob[1, 1], o)

# Contoh Soal 5.3: NaN pada pad kiri
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Mk = mask_kausal(3) + mask_dari_boolean(np.array([False, True, True]))
    assert np.all(np.isnan(softmax(Mk[0])))
    assert np.all(np.isfinite(softmax(Mk[1:])))

# Contoh Soal 5.4: cross-attention, query cahaya
qc = E[3] @ WQ
assert np.array_equal(qc, [0, 1]) and np.array_equal(K @ qc, [1, 0, 1])
Oc, Ac = attn(E[[0, 3]] @ WQ, K, V)
assert round(np.exp(1 / np.sqrt(2)) * 2 + 1, 4) == 5.0562
assert np.allclose(r4(Ac[1]), [0.4011, 0.1978, 0.4011])
assert np.allclose(r4(Oc[1]), [0.2033, 1.4011])
assert np.isclose(qc @ K[1], 0)
assert np.allclose(Ac[0], At[0])

# Contoh Soal 5.5: banyaknya pasangan
n, w = 1024, 128
assert n * (n + 1) // 2 == 524_800 == pola_mask("kausal", n).sum()
assert n * n == 1_048_576
assert n * w - w * (w - 1) // 2 == 122_944 == pola_mask("jendela", n, w=w).sum()
assert n * w == 131_072 and w * (w - 1) // 2 == 8_128
assert n - w == 896 and 122_944 + 896 == 123_840 == pola_mask(
    "global", n, w=w).sum()
assert round(100 * 122_944 / 524_800, 1) == 23.4
# Gambar 5.1: banyaknya pasangan n = 8
assert [pola_mask(k, 8).sum() for k in
        ("penuh", "kausal", "jendela", "global", "awalan", "dokumen")] == \
    [64, 36, 21, 26, 39, 21]
print("Bab 5: semua Contoh Soal cocok")
