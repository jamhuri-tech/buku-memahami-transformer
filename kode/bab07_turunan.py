"""Bab 7: turunan softmax dan attention dengan tangan, backward lengkap
satu lapisan data mini, cek gradien, dan saturasi.

Fungsi yang dipakai bab-bab berikutnya:
  jacobian_softmax(p)               diag(p) - p p^T
  vjp_softmax(A, dA)                A * (dA - sum(A * dA)) per baris
  attn_mundur(Q, K, V, A, dO, sk)   (dQ, dK, dV)
  maju_lapisan(par, masuk)          lintasan maju Bab 1, menyimpan antara
  mundur_lapisan(par, cache, y)     gradien semua parameter
"""
import numpy as np

from bab01_data import (E, MASUK, SASARAN, W1, W2, WK, WQ, WV,
                        mask_kausal, one_hot, softmax)


def jacobian_softmax(p):
    p = np.asarray(p, dtype=float)
    return np.diag(p) - np.outer(p, p)


def vjp_softmax(A, dA):
    """dS = J^T dA per baris, tanpa membentuk J."""
    return A * (dA - (A * dA).sum(-1, keepdims=True))


def attn_mundur(Q, K, V, A, dO, sk):
    dV = A.T @ dO
    dA = dO @ V.T
    dS = vjp_softmax(A, dA)
    dQ = dS @ K * sk
    dK = dS.T @ Q * sk
    return dQ, dK, dV


def parameter_mini():
    return dict(E=E.copy(), WQ=WQ.copy(), WK=WK.copy(), WV=WV.copy(),
                W1=W1.copy(), W2=W2.copy())


def maju_lapisan(par, masuk=MASUK):
    c = {"masuk": masuk}
    X = par["E"][masuk]
    n = len(masuk)
    Q, K, V = X @ par["WQ"], X @ par["WK"], X @ par["WV"]
    sk = 1 / np.sqrt(K.shape[1])
    A = softmax(Q @ K.T * sk + mask_kausal(n))
    H = X + A @ V
    U = H @ par["W1"]
    Z = np.maximum(U, 0)
    H2 = H + Z @ par["W2"]
    P = softmax(H2 @ par["E"].T)
    c.update(X=X, Q=Q, K=K, V=V, sk=sk, A=A, H=H, U=U, Z=Z, H2=H2, P=P)
    return c


def loss(c, sasaran=SASARAN):
    P = c["P"]
    return -np.mean(np.log(P[np.arange(len(sasaran)), sasaran]))


def mundur_lapisan(par, c, sasaran=SASARAN):
    n = len(sasaran)
    g = {}
    dLogit = (c["P"] - one_hot(sasaran, len(par["E"]))) / n
    g["E"] = dLogit.T @ c["H2"]                 # jalan keluaran
    dH2 = dLogit @ par["E"]
    # FFN: H2 = H + ReLU(H W1) W2
    g["W2"] = c["Z"].T @ dH2
    dZ = dH2 @ par["W2"].T
    dU = dZ * (c["U"] > 0)
    g["W1"] = c["H"].T @ dU
    dH = dH2 + dU @ par["W1"].T
    # residual: H = X + A V
    dX = dH.copy()
    dQ, dK, dV = attn_mundur(c["Q"], c["K"], c["V"], c["A"], dH, c["sk"])
    X = c["X"]
    g["WQ"], g["WK"], g["WV"] = X.T @ dQ, X.T @ dK, X.T @ dV
    dX += dQ @ par["WQ"].T + dK @ par["WK"].T + dV @ par["WV"].T
    np.add.at(g["E"], c["masuk"], dX)           # jalan masukan
    g["X"] = dX
    return g


def cek_beda_hingga(par, eps=1e-6):
    """Galat relatif maksimum gradien analitik lawan beda pusat."""
    g = mundur_lapisan(par, maju_lapisan(par))
    hasil = {}
    for k in ("E", "WQ", "WK", "WV", "W1", "W2"):
        num = np.zeros_like(par[k])
        for idx in np.ndindex(par[k].shape):
            p1 = {a: b.copy() for a, b in par.items()}
            p2 = {a: b.copy() for a, b in par.items()}
            p1[k][idx] += eps
            p2[k][idx] -= eps
            num[idx] = (loss(maju_lapisan(p1)) -
                        loss(maju_lapisan(p2))) / (2 * eps)
        hasil[k] = np.abs(num - g[k]).max() / max(np.abs(num).max(), 1e-12)
    return hasil


if __name__ == "__main__":
    import torch

    np.set_printoptions(precision=4, suppress=True)
    par = parameter_mini()
    c = maju_lapisan(par)
    g = mundur_lapisan(par, c)

    t = {k: torch.tensor(v, requires_grad=True) for k, v in par.items()}
    Xt = t["E"][MASUK]
    O = torch.nn.functional.scaled_dot_product_attention(
        (Xt @ t["WQ"])[None], (Xt @ t["WK"])[None], (Xt @ t["WV"])[None],
        is_causal=True)[0]
    H = Xt + O
    H2 = H + torch.relu(H @ t["W1"]) @ t["W2"]
    L = torch.nn.functional.cross_entropy(H2 @ t["E"].T,
                                          torch.tensor(SASARAN))
    L.backward()
    print("(1) Backward tangan lawan autograd PyTorch, data mini:")
    print(f"    loss = {loss(c):.4f} (PyTorch {L.item():.4f})")
    for k in ("WQ", "WK", "WV", "W1", "W2", "E"):
        beda = np.abs(t[k].grad.numpy() - g[k]).max()
        print(f"    {k:3s} selisih maks = {beda:.1e}")
    print("    dL/dWV =", str(g["WV"]).replace("\n", "\n            "))

    print("(2) Beda hingga pusat, eps = 1e-6, galat relatif maks:")
    print("    U = H W1 =", str(c["U"]).replace("\n", "\n               "))
    rng = np.random.default_rng(20261009)
    geser = {k: v + 0.01 * rng.normal(size=v.shape) for k, v in par.items()}
    a, b = cek_beda_hingga(par), cek_beda_hingga(geser)
    print("        data mini   digeser 0.01")
    for k in a:
        print(f"    {k:3s} {a[k]:9.1e} {b[k]:14.1e}")

    print("(3) Rata-rata ||J softmax||_F, 16 key, q, k ~ N(0, I):")
    print("     d_k   tanpa skala   dengan skala")
    for dk in (4, 16, 64, 256):
        nt, nd = [], []
        for _ in range(2000):
            s = rng.normal(size=(16, dk)) @ rng.normal(size=dk)
            nt.append(np.linalg.norm(jacobian_softmax(softmax(s))))
            nd.append(np.linalg.norm(jacobian_softmax(
                softmax(s / np.sqrt(dk)))))
        print(f"    {dk:4d} {np.mean(nt):13.4f} {np.mean(nd):14.4f}")
