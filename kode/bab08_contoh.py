"""Memeriksa setiap bilangan Contoh Soal Bab 8."""
import math

import numpy as np

from bab01_data import E, W1, W2, data_mini, lintasan_maju
from bab01_alur import banyak_parameter
from bab08_ffn import PHI, dgelu, dsilu, gelu, silu

r4 = lambda v: np.round(v, 4)
h = lintasan_maju(data_mini())

# teks: parameter FFN dan attention
for d in (64, 768):
    r = banyak_parameter(d, 1, 4 * d, 1, 0, ln_akhir=False)
    assert r["ffn"] == 8 * d * d + 5 * d and r["attn"] == 4 * d * d + 4 * d

# Contoh Soal 8.1: FFN sebagai memori
assert np.array_equal(W1.T, E)
h3 = h["H"][2]
assert np.allclose(r4(h3), [0.856, 2.576])
kec = h3 @ W1
assert np.allclose(np.round(kec, 3), [0.856, 2.576, 3.432, 1.720])
f3 = sum(max(c, 0) * W2[j] for j, c in enumerate(kec))
assert np.allclose(np.round(f3, 3), [-1.288, 0.428])
assert np.allclose(f3, h["H2"][2] - h["H"][2])
assert np.array_equal(h["H"][0] @ W1, [2, 1, 3, -1])

# Contoh Soal 8.2: GELU dan SiLU
phi1 = math.exp(-0.5) / math.sqrt(2 * math.pi)
assert round(float(PHI(1)), 4) == 0.8413 and round(phi1, 4) == 0.2420
assert round(float(gelu(1.0)), 4) == 0.8413
assert round(float(gelu(-1.0)), 4) == -0.1587
assert round(float(dgelu(1.0)), 4) == 1.0833 == round(0.8413 + 0.2420, 4)
assert np.isclose(dgelu(0.0), 0.5) and np.isclose(dsilu(0.0), 0.5)
assert round(float(silu(1.0)), 4) == 0.7311
assert np.isclose(gelu(1.0) - gelu(-1.0), 1)
# teks: hampiran tanh selisih di bawah 1e-3
from bab08_ffn import gelu_tanh
xx = np.linspace(-6, 6, 2001)
assert np.abs(gelu(xx) - gelu_tanh(xx)).max() < 1e-3

# Contoh Soal 8.3: SwiGLU
assert 2 * 768 * 3072 == 4_718_592 and 4_718_592 / 2304 == 2048
assert 3 * 768 * 2048 == 4_718_592 and 8 * 768 / 3 == 2048

# Contoh Soal 8.4: jaringan skalar
assert round(0.5**10, 6) == 0.000977 and 0.05**10 < 1e-12
assert round(1.5**10, 2) == 57.67 and round(1.05**10, 3) == 1.629
assert round(math.exp(0.5), 3) == 1.649

# Contoh Soal 8.5: atribusi logit
x1, o1, f1 = data_mini()[0], h["O"][0], h["F"][0]
assert np.allclose(o1, [1, 1]) and np.allclose(f1, [-0.5, 0.5])
assert np.allclose(x1 + o1 + f1, [1.5, 1.5])
itu, ilmu = E[2], E[1]
assert [x1 @ itu, o1 @ itu, f1 @ itu] == [1, 2, 0]
assert [x1 @ ilmu, o1 @ ilmu, f1 @ ilmu] == [0, 1, 0.5]
print("Bab 8: semua Contoh Soal cocok")
