"""Bab 8: FFN sebagai memori key-value, fungsi aktivasi, SwiGLU, dan
sambungan residual pada jaringan dalam.

Fungsi yang dipakai bab-bab berikutnya:
  gelu(x), gelu_tanh(x), silu(x)      aktivasi dan turunannya (d*)
  ffn(H, W1, W2, b1, b2, akt)         FFN dua lapisan
  swiglu(H, W, Vg, W2)                (SiLU(H W) * (H Vg)) W2
  norma_gradien(N, residual, ...)     ||dL/dx_0|| jaringan sedalam N
"""
import math

import numpy as np

PHI = np.vectorize(lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2))))


def relu(x):
    return np.maximum(x, 0)


def gelu(x):
    return x * PHI(x)


def dgelu(x):
    return PHI(x) + x * np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def silu(x):
    return x / (1 + np.exp(-x))


def dsilu(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 + x * (1 - s))


def ffn(H, W1, W2, b1=0, b2=0, akt=relu):
    return akt(H @ W1 + b1) @ W2 + b2


def swiglu(H, W, Vg, W2):
    return (silu(H @ W) * (H @ Vg)) @ W2


def norma_gradien(N, residual, d=64, skala=1.0, benih=20261009):
    """x_{l+1} = x_l + F(x_l) (atau F(x_l) saja), F = ReLU(x W) U.
    Mengembalikan ||dL/dx_0|| / ||dL/dx_N|| untuk gradien hulu acak."""
    rng = np.random.default_rng(benih)
    x = rng.normal(size=d)
    lap = []
    for _ in range(N):
        W = rng.normal(size=(d, d)) * np.sqrt(2 / d)
        U = rng.normal(size=(d, d)) * np.sqrt(1 / d) * skala
        u = x @ W
        f = np.maximum(u, 0) @ U
        lap.append((W, U, u))
        x = x + f if residual else f
    g = rng.normal(size=d)
    g0 = np.linalg.norm(g)
    for W, U, u in reversed(lap):
        gf = ((g @ U.T) * (u > 0)) @ W.T
        g = g + gf if residual else gf
    return np.linalg.norm(g) / g0


if __name__ == "__main__":
    import torch

    from bab01_data import data_mini, lintasan_maju
    from bab01_data import W1 as W1m, W2 as W2m
    from bab01_alur import banyak_parameter

    np.set_printoptions(precision=4, suppress=True)
    x = np.array([-2.0, -1, -0.5, 0, 0.5, 1, 2])
    tx = torch.tensor(x)
    print("(1) Aktivasi:")
    print("    x    " + "".join(f"{v:7.1f}" for v in x))
    for nama, f in (("ReLU", relu), ("GELU", gelu), ("SiLU", silu)):
        print(f"    {nama:4s} " + "".join(f"{v + 0:7.3f}" for v in f(x)))
    xx = np.linspace(-6, 6, 2001)
    print("    maks |GELU - hampiran tanh| di [-6, 6] ="
          f" {np.abs(gelu(xx) - gelu_tanh(xx)).max():.1e}")
    beda = max(np.abs(torch.nn.functional.gelu(tx).numpy() - gelu(x)).max(),
               np.abs(torch.nn.functional.gelu(tx, approximate="tanh")
                      .numpy() - gelu_tanh(x)).max(),
               np.abs(torch.nn.functional.silu(tx).numpy() - silu(x)).max())
    print(f"    selisih maks dengan PyTorch = {beda:.1e}")

    h = lintasan_maju(data_mini())
    print("(2) FFN data mini sebagai memori: z = ReLU(H W1)")
    print("    kolom W1 (key) :", [tuple(c) for c in W1m.T.astype(int)])
    print("    z =", str(h["Z"]).replace("\n", "\n        "))
    print("    unit aktif:",
          [list(np.flatnonzero(r > 0) + 1) for r in h["Z"]])

    print("(3) ||dL/dx_0|| / ||dL/dx_N||, d = 64, F = ReLU(xW)U:")
    print("          tanpa residual          residual")
    print("      N   U x 1     U x 0.5     U x 1     U/sqrt(2N)")
    for N in (2, 8, 32, 64):
        a = norma_gradien(N, False)
        b = norma_gradien(N, False, skala=0.5)
        c = norma_gradien(N, True)
        e = norma_gradien(N, True, skala=1 / np.sqrt(2 * N))
        print(f"    {N:3d} {a:9.2e} {b:11.2e} {c:9.2e} {e:9.2f}")

    print("(4) Bagian FFN dari parameter satu lapisan (d_ff = 4d):")
    for d in (64, 768, 4096):
        r = banyak_parameter(d, 1, 4 * d, 1, 0, ln_akhir=False)
        print(f"    d = {d:4d}: attention {r['attn']:>11,d},"
              f" FFN {r['ffn']:>12,d}, {r['ffn'] / r['lapisan']:.3f}")
