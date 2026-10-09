"""Bab 2: gradien embedding dengan weight tying, dicocokkan dengan
autograd PyTorch, dan kemiripan kosinus embedding data mini.

Fungsi yang dipakai bab-bab berikutnya:
  kosinus(a, b)   kemiripan kosinus dua vektor
"""
import numpy as np

from bab01_data import (E, KOSAKATA, MASUK, SASARAN, W1, W2, WK, WQ, WV,
                        data_mini, lintasan_maju, one_hot)


def kosinus(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def gradien_keluaran(h):
    """Bagian dL/dE dari logit Z = H' E^T: G_Z^T H'."""
    GZ = (h["P"] - one_hot(SASARAN)) / len(SASARAN)
    return GZ.T @ h["H2"]


if __name__ == "__main__":
    import torch

    np.set_printoptions(precision=4, suppress=True)
    h = lintasan_maju(data_mini())
    Gk = gradien_keluaran(h)

    # Autograd: E dipakai dua kali, sebagai masukan dan sebagai keluaran.
    tE = torch.tensor(E, requires_grad=True)
    t = {k: torch.tensor(v) for k, v in
         dict(WQ=WQ, WK=WK, WV=WV, W1=W1, W2=W2).items()}

    def maju(Emasuk, Ekeluar):
        X = Emasuk[MASUK]
        Q, K, V = X @ t["WQ"], X @ t["WK"], X @ t["WV"]
        O = torch.nn.functional.scaled_dot_product_attention(
            Q[None], K[None], V[None], is_causal=True)[0]
        H = X + O
        H2 = H + torch.relu(H @ t["W1"]) @ t["W2"]
        return torch.nn.functional.cross_entropy(
            H2 @ Ekeluar.T, torch.tensor(SASARAN))

    maju(tE, tE).backward()
    total = tE.grad.numpy()
    # bagian masukan saja: E keluaran dibekukan
    tE2 = torch.tensor(E, requires_grad=True)
    maju(tE2, torch.tensor(E)).backward()
    masuk = tE2.grad.numpy()
    print("(3) Gradien dL/dE data mini (baris = token):")
    f2 = lambda a: f"{a[0] + 0:7.4f} {a[1] + 0:7.4f}"
    print(f"    {'token':6s} {'dari masukan':>15s} {'dari keluaran':>15s}"
          f" {'total':>15s}")
    for v in range(4):
        print(f"    {KOSAKATA[v]:6s} {f2(masuk[v])} {f2(Gk[v])}"
              f" {f2(total[v])}")
    print("    selisih maks (total - masukan - keluaran) ="
          f" {np.abs(total - masuk - Gk).max():.1e}")
    print("    jumlah baris bagian keluaran:", f2(Gk.sum(0)))
