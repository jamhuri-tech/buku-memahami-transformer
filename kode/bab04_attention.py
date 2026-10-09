"""Bab 4: scaled dot-product attention, skala 1/sqrt(d_k), softmax
yang stabil, entropi attention, dan hubungan dengan regresi kernel.

Fungsi yang dipakai bab-bab berikutnya:
  attn(Q, K, V, M=None, skala=None)  keluaran dan bobot attention
  entropi(A)                          entropi setiap baris bobot
  nadaraya_watson(x, xs, ys, h)       regresi kernel Gauss
"""
import numpy as np

from bab01_data import softmax


def attn(Q, K, V, M=None, skala=None):
    """softmax(Q K^T * skala + M) V; skala bawaan 1/sqrt(d_k)."""
    if skala is None:
        skala = 1 / np.sqrt(K.shape[-1])
    S = Q @ np.swapaxes(K, -1, -2) * skala
    if M is not None:
        S = S + M
    A = softmax(S)
    return A @ V, A


def softmax_naif(z):
    e = np.exp(np.asarray(z, dtype=float))
    return e / e.sum()


def entropi(A):
    """-sum a log a per baris, dengan 0 log 0 = 0."""
    A = np.asarray(A, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(A > 0, A * np.log(A), 0.0)
    return -L.sum(axis=-1)


def nadaraya_watson(x, xs, ys, h):
    """sum_s K(x, x_s) y_s / sum_s K(x, x_s), K Gauss dengan lebar h."""
    w = softmax(-(np.asarray(x)[:, None] - xs[None, :]) ** 2 / (2 * h * h))
    return w @ ys


def simulasi_skala(dk, n=16, ulang=4000, rng=None):
    """Varians q.k dan rata-rata bobot terbesar, entri q, k ~ N(0, 1)."""
    q = rng.normal(size=(ulang, 1, dk))
    k = rng.normal(size=(ulang, n, dk))
    s = (q @ np.swapaxes(k, 1, 2))[:, 0, :]
    tanpa = softmax(s).max(axis=1).mean()
    dengan = softmax(s / np.sqrt(dk)).max(axis=1).mean()
    return s.var(), tanpa, dengan


if __name__ == "__main__":
    import warnings

    import torch
    import torch.nn.functional as Fn

    from bab01_data import WK, WQ, WV, data_mini, mask_kausal

    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(20261009)
    print("(1) Entri q, k ~ N(0, 1), 16 key, 4000 ulangan:")
    print("     d_k   Var(q.k)   bobot maks tanpa skala   dengan skala")
    for dk in (4, 16, 64, 256):
        v, a, b = simulasi_skala(dk, rng=rng)
        print(f"    {dk:4d} {v:10.2f} {a:24.3f} {b:14.3f}")

    X = data_mini()
    Q, K, V = X @ WQ, X @ WK, X @ WV
    O, A = attn(Q, K, V)
    tO = Fn.scaled_dot_product_attention(*(torch.tensor(m)[None]
                                           for m in (Q, K, V)))[0]
    print("(2) Data mini tanpa mask (encoder):")
    print("    A =", str(A).replace("\n", "\n        "))
    print("    O =", str(O).replace("\n", "\n        "))
    print(f"    PyTorch: selisih maks O = {np.abs(tO.numpy() - O).max():.1e}")
    Q5 = rng.normal(size=(5, 8))
    K5 = rng.normal(size=(5, 8))
    V5 = rng.normal(size=(5, 3))
    O5, _ = attn(Q5, K5, V5, M=mask_kausal(5))
    t5 = Fn.scaled_dot_product_attention(
        *(torch.tensor(m)[None] for m in (Q5, K5, V5)), is_causal=True)[0]
    print(f"    acak n = 5, d_k = 8, kausal: selisih maks ="
          f" {np.abs(t5.numpy() - O5).max():.1e}")

    print("(3) Softmax untuk z = (1000, 1001, 1002):")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("    langsung exp(z)/jumlah :", softmax_naif([1000, 1001, 1002]))
    print("    dikurangi maks dahulu  :", softmax([1000, 1001, 1002]))
