"""Bab 3: attention tanpa posisi, encoding sinusoidal, RoPE, dan ALiBi.

Fungsi yang dipakai bab-bab berikutnya:
  pe_sinus(n, d, basis)     encoding posisi sinusoidal, n x d
  sudut_rope(n, d, basis)   sudut p * theta_i, n x d/2
  rope(X, sudut)            memutar setiap pasangan koordinat baris X
  bias_alibi(n, m)          matriks -m (t - s) untuk s <= t
  attention(X, kausal, ...) attention satu head data mini dengan posisi
"""
import numpy as np

from bab01_data import WK, WQ, WV, data_mini, mask_kausal, softmax


def pe_sinus(n, d, basis=10000.0):
    """PE[p, 2i] = sin(p w_i), PE[p, 2i+1] = cos(p w_i), w_i = basis^(-2i/d)."""
    p = np.arange(n)[:, None]
    w = basis ** (-np.arange(0, d, 2) / d)
    PE = np.zeros((n, d))
    PE[:, 0::2] = np.sin(p * w)
    PE[:, 1::2] = np.cos(p * w)
    return PE


def sudut_rope(n, d, basis=10000.0):
    theta = basis ** (-np.arange(0, d, 2) / d)
    return np.arange(n)[:, None] * theta


def rope(X, sudut):
    """Pasangan (x_2i, x_2i+1) diputar sejauh sudut[p, i]."""
    c, s = np.cos(sudut), np.sin(sudut)
    a, b = X[:, 0::2], X[:, 1::2]
    Y = np.empty_like(X, dtype=float)
    Y[:, 0::2] = a * c - b * s
    Y[:, 1::2] = a * s + b * c
    return Y


def bias_alibi(n, m):
    t = np.arange(n)
    return -m * (t[:, None] - t[None, :]).astype(float)


def attention(X, kausal=True, posisi=None, sudut=None, bias=None):
    """Satu head dengan bobot data mini; posisi: PE yang dijumlahkan."""
    if posisi is not None:
        X = X + posisi
    Q, K, V = X @ WQ, X @ WK, X @ WV
    if sudut is not None:
        Q, K = rope(Q, sudut), rope(K, sudut)
    S = Q @ K.T / np.sqrt(K.shape[1])
    if bias is not None:
        S = S + bias
    if kausal:
        S = S + mask_kausal(len(X))
    A = softmax(S)
    return A, A @ V


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    X = data_mini()
    perm = [1, 2, 0]                      # ilmu itu <awal>
    _, O = attention(X, kausal=False)
    _, Op = attention(X[perm], kausal=False)
    print("(1) Tanpa posisi dan tanpa mask: Attn(PX) = P Attn(X)?")
    print("    selisih maks =", f"{np.abs(Op - O[perm]).max():.1e}")
    _, O = attention(X, kausal=True)
    _, Op = attention(X[perm], kausal=True)
    print("    dengan causal mask: selisih maks =",
          f"{np.abs(Op - O[perm]).max():.4f}")

    rng = np.random.default_rng(20261009)
    d, n = 64, 512
    PE = pe_sinus(n, d)
    G = PE @ PE.T
    beda = max(abs(G[p, p + k] - G[0, k]) for k in range(50)
               for p in range(0, n - 50, 37))
    print("(2) Sinusoidal d = 64: PE(p).PE(p+k) hanya bergantung pada k")
    print(f"    selisih maks atas p = {beda:.1e}")
    print("    PE(p).PE(p+k) untuk k = 0, 1, 2, 5, 10, 50, 100:")
    print("    ", np.round([G[0, k] for k in (0, 1, 2, 5, 10, 50, 100)], 2))

    q = rng.normal(size=(1, d))
    k = rng.normal(size=(1, d))
    sd = sudut_rope(n, d)
    skor = lambda p, s: (rope(q, sd[[p]]) @ rope(k, sd[[s]]).T).item()
    print("(3) RoPE d = 64: skor q_p . k_s hanya bergantung pada s - p")
    print(f"    (p, s) = (3, 10): {skor(3, 10):.4f},"
          f" (103, 110): {skor(103, 110):.4f}")
    # bentuk kompleks: pasangan sebagai bilangan kompleks dikali e^{i sudut}
    z = (q[:, 0::2] + 1j * q[:, 1::2]) * np.exp(1j * sd[[5]])
    y = rope(q, sd[[5]])
    print("    cocok dengan perkalian kompleks:",
          np.allclose(z.real, y[:, 0::2]) and np.allclose(z.imag,
                                                          y[:, 1::2]))
    print("    norma q tidak berubah:",
          np.isclose(np.linalg.norm(y), np.linalg.norm(q)))
