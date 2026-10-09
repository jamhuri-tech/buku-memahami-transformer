"""Memeriksa setiap bilangan Contoh Soal Bab 12."""
import math

import numpy as np
import torch

from bab12_optimisasi import Adam, laju_belajar, potong_norma

# Contoh Soal 12.1: dua langkah Adam
a = Adam(lr=0.1, eps=0.0)
th = a.langkah(np.array([0.0]), np.array([2.0]))
assert np.isclose(a.m[0], 0.2) and np.isclose(a.v[0], 0.004)
assert np.isclose(th[0], -0.1)
th = a.langkah(th, np.array([-1.0]))
assert np.isclose(a.m[0], 0.08) and np.isclose(a.v[0], 0.004996)
assert np.isclose(1 - 0.9**2, 0.19) and np.isclose(1 - 0.999**2, 0.001999)
mh, vh = 0.08 / 0.19, 0.004996 / 0.001999
assert round(mh, 6) == 0.421053 and round(vh, 6) == 2.49925
assert round(math.sqrt(vh), 6) == 1.580902
assert round(th[0], 6) == -0.126634
# invarian skala
b = Adam(lr=0.1, eps=0.0)
t2 = b.langkah(np.array([0.0]), np.array([2000.0]))
t2 = b.langkah(t2, np.array([-1000.0]))
assert np.isclose(t2[0], th[0])

# Contoh Soal 12.2: AdamW lawan L2
w = torch.tensor([1.0], requires_grad=True)
o = torch.optim.AdamW([w], lr=0.01, weight_decay=0.1)
w.grad = torch.tensor([0.0])
o.step()
assert np.isclose(w.item(), 0.999)
c = Adam(lr=0.01, eps=0.0)
assert np.isclose(c.langkah(np.array([1.0]), np.array([0.1]))[0], 0.99)
c = Adam(lr=0.01, eps=0.0)
assert np.isclose(c.langkah(np.array([1000.0]), np.array([100.0]))[0],
                  1000 - 0.01)
w = torch.tensor([1000.0], requires_grad=True)
o = torch.optim.AdamW([w], lr=0.01, weight_decay=0.1)
w.grad = torch.tensor([0.0])
o.step()
assert np.isclose(w.item(), 999.0)

# Contoh Soal 12.3: jadwal
assert np.isclose(laju_belajar(49, 1e-3, 100, 1000), 5e-4)
assert np.isclose(laju_belajar(99, 1e-3, 100, 1000), 1e-3)
assert np.isclose(laju_belajar(550, 1e-3, 100, 1000), 5e-4)
assert round(laju_belajar(775, 1e-3, 100, 1000), 7) == 1.464e-4
assert round(1 + math.cos(3 * math.pi / 4), 4) == 0.2929

# Contoh Soal 12.4: clipping
g, n = potong_norma([np.array([3.0]), np.array([4.0])], 1.0)
assert n == 5 and np.isclose(g[0][0], 0.6) and np.isclose(g[1][0], 0.8)
p1 = torch.nn.Parameter(torch.zeros(1))
p2 = torch.nn.Parameter(torch.zeros(1))
p1.grad, p2.grad = torch.tensor([3.0]), torch.tensor([4.0])
torch.nn.utils.clip_grad_norm_([p1, p2], 1.0)
assert np.isclose(p1.grad.item(), 0.6, atol=1e-6)
# latihan 2: tanpa koreksi bias
assert round(0.1 / math.sqrt(0.001), 2) == 3.16
print("Bab 12: semua Contoh Soal cocok")
