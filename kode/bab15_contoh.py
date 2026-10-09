"""Memeriksa setiap bilangan Contoh Soal Bab 15."""
import math

import numpy as np
import torch

from bab15_finetuning import LoRA, Klasifikasi, model_dasar, pasang_lora

# Contoh Soal 15.1: gradien LoRA r = 1
dW = np.array([[0.716, 0], [0.860, 0]])
A = np.array([[1.0], [1.0]])
B = np.zeros((1, 2))
dA, dB = dW @ B.T, A.T @ dW
assert not dA.any() and np.allclose(dB, [[1.576, 0]])
B = B - 0.1 * dB
assert np.allclose(B, [[-0.1576, 0]])
WV = np.array([[1.0, 1], [-1, 1]])
assert np.allclose(WV + A @ B, [[0.8424, 1], [-1.1576, 1]])
assert np.linalg.matrix_rank(A @ B) == 1
# sama dengan autograd melalui parametrisasi LoRA (bentuk PyTorch B A)
L = LoRA(2, 2, r=1, alfa=1)
with torch.no_grad():
    L.A.copy_(torch.tensor(A.T))
W = torch.tensor(WV.T)
G = torch.tensor(dW.T)
(L(W) * G).sum().backward()
assert np.allclose(L.B.grad.numpy().T, dB)

# Contoh Soal 15.2: parameter LoRA
blok = 4 * (128 + 384) + 4 * (128 + 128) + 4 * (128 + 512) + 4 * (512 + 128)
assert blok == 8192 and 2048 + 1024 + 2560 + 2560 == 8192
assert 4 * blok == 32_768 and 128 * 9 + 9 == 1161 and 32_768 + 1161 == 33_929
m = Klasifikasi(pasang_lora(model_dasar(), 4), 9)
assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 33_929
assert 8 * 8192 == 65_536 and 4096**2 == 16_777_216
assert round(100 * 65_536 / 16_777_216, 2) == 0.39
assert round(821_001 / 33_929, 1) == 24.2
assert 819_840 + 1161 == 821_001

# Contoh Soal 15.3: galat baku dan z
se = lambda p: math.sqrt(p * (1 - p) / 900)
assert round(se(0.334), 4) == 0.0157 and round(se(0.343), 4) == 0.0158
assert round(math.sqrt(0.0157**2 + 0.0158**2), 4) == 0.0223
assert round(0.009 / 0.0223, 2) == 0.40
assert round(math.sqrt(se(0.334) ** 2 + se(0.280) ** 2), 4) == 0.0217
assert round(0.054 / 0.0217, 2) == 2.49
print("Bab 15: semua Contoh Soal cocok")
