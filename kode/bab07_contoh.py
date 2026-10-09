"""Memeriksa setiap bilangan Contoh Soal Bab 7."""
import numpy as np

from bab01_data import WK, WQ, WV, data_mini, lintasan_maju
from bab07_turunan import (attn_mundur, jacobian_softmax, maju_lapisan,
                           parameter_mini, vjp_softmax)

r4 = lambda v: np.round(v, 4)
X = data_mini()
Q, K, V = X @ WQ, X @ WK, X @ WV
A = lintasan_maju(X)["A"]

# Contoh Soal 7.1: Jacobian
assert np.allclose(jacobian_softmax([0.5, 0.5]), [[.25, -.25], [-.25, .25]])
J = jacobian_softmax(A[2])
a = r4(A[2])
assert np.allclose(a, [0.14, 0.284, 0.576])
assert np.allclose(r4(np.diag(J)), [0.1204, 0.2033, 0.2442])
assert np.allclose(r4([J[0, 1], J[0, 2], J[1, 2]]), [-0.0398, -0.0807, -0.1636])
assert round(0.14 * 0.86, 4) == 0.1204 and round(0.284 * 0.716, 4) == 0.2033
assert round(0.576 * 0.424, 4) == 0.2442
assert round(0.1204 - 0.0398 - 0.0807, 4) == -0.0001
assert np.allclose(J.sum(1), 0) and np.allclose(J, J.T)
assert np.all(np.linalg.eigvalsh(J) > -1e-12)
p = np.array([0.98, 0.01, 0.01])
assert round(np.trace(jacobian_softmax(p)), 4) == 0.0394
assert np.isclose(np.trace(jacobian_softmax(np.full(16, 1 / 16))), 0.9375)
# Persamaan PSD: u^T J u = varians
u = np.array([1.0, -2, 0.5])
assert np.isclose(u @ J @ u, A[2] @ u**2 - (A[2] @ u) ** 2)

# Contoh Soal 7.2: dA dan dS
dO = np.zeros((3, 2))
dO[2, 0] = 1
dA = dO @ V.T
assert np.allclose(dA[2], [1, -1, 0]) and not dA[:2].any()
assert np.isclose(round(A[2] @ dA[2], 4), -0.1440)
dS = vjp_softmax(A, dA)
assert np.allclose(r4(dS[2]), [0.1602, -0.2431, 0.0829])
assert np.isclose(dS[2].sum(), 0) and not dS[:2].any()
assert np.allclose(J @ dA[2], dS[2])
assert round(0.1204 + 0.0398, 4) == 0.1602
assert np.allclose(r4(a * (np.array([1, -1, 0]) + 0.144)), [0.1602, -0.2431, 0.0829])

# Contoh Soal 7.3: dQ, dK, dV, dWV
dQ, dK, dV = attn_mundur(Q, K, V, A, dO, 1 / np.sqrt(2))
assert np.allclose(r4(dS[2] @ K), [-0.1602, 0.2431])
assert np.allclose(r4(dQ), [[0, 0], [0, 0], [-0.1133, 0.1719]])
assert np.allclose(r4(dK), [[0.2265, 0.1133], [-0.3438, -0.1719],
                            [0.1173, 0.0586]])
assert np.allclose(r4(dV), [[0.14, 0], [0.284, 0], [0.576, 0]])
dWV = X.T @ dV
assert np.allclose(r4(dWV), [[0.716, 0], [0.860, 0]])
# cocok dengan autograd untuk L = o_31
import torch
t = {k: torch.tensor(v, requires_grad=True) for k, v in
     dict(Q=Q, K=K, V=V).items()}
o = torch.nn.functional.scaled_dot_product_attention(
    t["Q"][None], t["K"][None], t["V"][None], is_causal=True)[0]
o[2, 0].backward()
assert np.allclose(t["Q"].grad.numpy(), dQ)
assert np.allclose(t["K"].grad.numpy(), dK)
assert np.allclose(t["V"].grad.numpy(), dV)

# Contoh Soal 7.4: lipatan ReLU
f = lambda x: max(x, 0.0)
eps = 1e-6
assert np.isclose((f(eps) - f(-eps)) / (2 * eps), 0.5)
assert np.isclose((f(1e-3 + eps) - f(1e-3 - eps)) / (2 * eps), 1)
c = maju_lapisan(parameter_mini())
assert c["U"][1, 0] == 0
ut = torch.tensor(0.0, requires_grad=True)
torch.relu(ut).backward()
assert ut.grad.item() == 0
print("Bab 7: semua Contoh Soal cocok")
