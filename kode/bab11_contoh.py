"""Memeriksa setiap bilangan Contoh Soal Bab 11."""
import numpy as np
import torch

from bab01_data import SASARAN, data_mini, lintasan_maju, loss_ce
from bab11_objektif import ce_halus

r4 = lambda v: np.round(v, 4)
h = lintasan_maju(data_mini())
L = loss_ce(h["P"], SASARAN)

# Contoh Soal 11.1: perplexity dan bit
assert round(np.exp(L), 4) == 5.4023 and round(L / np.log(2), 4) == 2.4336
assert round(np.log(2), 4) == 0.6931 and np.log2(4) == 2
assert round(2 ** (L / np.log(2)), 4) == 5.4023

# Contoh Soal 11.2: H, CE, KL
p = np.array([0.5, 0.25, 0.25, 0])
q = np.full(4, 0.25)
H = -np.sum(p[p > 0] * np.log(p[p > 0]))
CE = -np.sum(p * np.log(q))
KL = np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0]))
assert round(0.5 * np.log(2), 4) == 0.3466 == round(0.25 * np.log(4), 4)
assert round(H, 4) == 1.0397 and round(CE, 4) == 1.3863
assert round(KL, 4) == 0.3466 and np.isclose(H + KL, CE)

# Contoh Soal 11.3: loss mask
l1 = -np.log(h["P"][np.arange(3), SASARAN])
assert np.allclose(r4(l1), [1.9028, 2.4197, 0.7380])
jml = 1.9028 + 2.4197 + 0.7380 + 2.0
assert round(jml, 4) == 7.0605 and round(jml / 4, 4) == 1.7651
assert round(jml / 6, 4) == 1.1767
assert round((1.6868 + 2.0) / 2, 4) == 1.8434
t = torch.tensor([[1.9028, 2.4197, 0.7380], [2.0, 0, 0]])
m = torch.tensor([[1.0, 1, 1], [1, 0, 0]])
assert round(((t * m).sum() / m.sum()).item(), 4) == 1.7651

# Contoh Soal 11.4: MLM
assert np.isclose(0.15 * 0.8, 0.12) and np.isclose(0.15 * 0.1, 0.015)
assert np.isclose(0.15 * 512, 76.8) and round(76.8 / 511, 2) == 0.15

# Contoh Soal 11.5: label smoothing
Z2 = h["logit"][1]
lse = np.log(np.exp(Z2).sum())
assert round(np.exp(Z2).sum(), 4) == 30.5608 and round(lse, 4) == 3.4197
nlp = lse - Z2
assert np.allclose(r4(nlp), [4.4197, 1.4197, 2.4197, 0.4197])
assert round(r4(nlp).sum(), 4) == 8.6788 and round(nlp.mean(), 4) == 2.1697
assert round(0.9 * 2.4197, 4) == 2.1777 and round(0.1 * 2.1697, 4) == 0.2170
assert round(ce_halus(Z2[None], [2], 0.1), 4) == 2.3947
assert round(np.log(0.925 / 0.025), 4) == 3.6109 and 0.925 / 0.025 == 37
lt = torch.nn.functional.cross_entropy(torch.tensor(Z2[None]),
                                       torch.tensor([2]), label_smoothing=0.1)
assert round(lt.item(), 4) == 2.3947
print("Bab 11: semua Contoh Soal cocok")
