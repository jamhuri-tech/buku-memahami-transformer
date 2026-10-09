"""Memeriksa setiap bilangan Contoh Soal Bab 16."""
import numpy as np
import torch

from bab16_luarteks import ViTMini, ke_patch

# Contoh Soal 16.1: banyaknya token
assert 64 // 4 == 16 and 2**2 * 1 == 4
assert 224**2 // 16**2 == 196 == 14**2 and 16**2 * 3 == 768
assert 197**2 == 38_809 and (224 // 8) ** 2 == 784 and 785**2 == 616_225
assert round(616_225 / 38_809) == 16
X = np.arange(64.0).reshape(1, 8, 8)
P = ke_patch(X, 2)
assert P.shape == (1, 16, 4) and list(P[0, 0]) == [0, 1, 8, 9]

# Contoh Soal 16.2: invarian [CLS] tanpa posisi
torch.manual_seed(0)
m = ViTMini(16, 4, posisi=False).eval()
x = torch.randn(3, 16, 4)
perm = torch.randperm(16)
with torch.no_grad():
    assert torch.allclose(m(x), m(x[:, perm]), atol=1e-5)
m2 = ViTMini(16, 4, posisi=True).eval()
with torch.no_grad():
    assert not torch.allclose(m2(x), m2(x[:, perm]), atol=1e-5)

# Contoh Soal 16.3: MSE naif AR(2)
g0, r1 = 1.5, 1 / 3
assert np.isclose(2 * g0 * (1 - r1), 2)

# Contoh Soal 16.4: patch deret
assert 512 // 16 == 32 and 32**2 == 1024 and 512**2 == 262_144
assert 262_144 // 1024 == 256 == 16**2
# teks: selisih Transformer terhadap AR(2)
assert round(1.0696 / 0.9931 - 1, 2) == 0.08
print("Bab 16: semua Contoh Soal cocok")
