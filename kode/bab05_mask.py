"""Bab 5: causal mask, padding mask, cross-attention, dan pola mask.

Fungsi yang dipakai bab-bab berikutnya:
  mask_dari_boolean(B)        True = boleh dilihat -> 0, False -> -inf
  mask_padding(panjang, n)    mask key untuk batch dengan pad di kanan
  pola_mask(nama, n, ...)     boolean n x n untuk beberapa pola
"""
import numpy as np

from bab01_data import WK, WQ, WV, E, data_mini, mask_kausal, softmax
from bab04_attention import attn


def mask_dari_boolean(B):
    return np.where(B, 0.0, -np.inf)


def mask_padding(panjang, n):
    """Batch x 1 x n: key ke-s boleh dilihat bila s < panjang."""
    B = np.arange(n)[None, :] < np.asarray(panjang)[:, None]
    return mask_dari_boolean(B)[:, None, :]


def pola_mask(nama, n, w=3, awalan=3, dokumen=(3, 5)):
    t = np.arange(n)
    kausal = t[None, :] <= t[:, None]
    if nama == "penuh":
        return np.ones((n, n), bool)
    if nama == "kausal":
        return kausal
    if nama == "jendela":
        return kausal & (t[:, None] - t[None, :] < w)
    if nama == "awalan":
        return kausal | (t[None, :] < awalan)
    if nama == "dokumen":
        doc = np.repeat(np.arange(len(dokumen)), dokumen)
        return kausal & (doc[:, None] == doc[None, :])
    if nama == "global":
        B = kausal & (t[:, None] - t[None, :] < w)
        B[:, 0] = True
        return B & kausal
    raise ValueError(nama)


if __name__ == "__main__":
    import warnings

    import torch
    import torch.nn.functional as Fn

    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(20261009)

    n, d = 6, 4
    Q, K, V = (rng.normal(size=(n, d)) for _ in range(3))
    O, _ = attn(Q, K, V, M=mask_kausal(n))
    beda = max(np.abs(attn(Q[:t], K[:t], V[:t], M=mask_kausal(t))[0][-1]
                      - O[t - 1]).max() for t in range(1, n + 1))
    O2, _ = attn(Q, K, V)
    beda2 = max(np.abs(attn(Q[:t], K[:t], V[:t])[0][-1]
                       - O2[t - 1]).max() for t in range(1, n + 1))
    print("(1) Posisi t: barisan penuh lawan awalan 1..t:")
    print(f"    dengan causal mask: selisih maks = {beda:.1e}")
    print(f"    tanpa mask        : selisih maks = {beda2:.4f}")

    # batch: <awal> ilmu itu | <awal> itu <pad>
    Xb = np.stack([E[[0, 1, 2]], np.r_[E[[0, 2]], [[9.0, 9.0]]]])
    Qb, Kb, Vb = Xb @ WQ, Xb @ WK, Xb @ WV
    M = mask_kausal(3)[None] + mask_padding([3, 2], 3)
    Ob, Ab = attn(Qb, Kb, Vb, M=M)
    boleh = torch.tensor(np.isfinite(M))
    tO = Fn.scaled_dot_product_attention(
        *(torch.tensor(m) for m in (Qb, Kb, Vb)), attn_mask=boleh)
    print("(2) Batch dua kalimat, causal + padding mask:")
    print("    A kalimat 2 =", str(Ab[1]).replace("\n", "\n                  "))
    print(f"    PyTorch: selisih maks O (baris bukan pad) ="
          f" {np.abs(tO.numpy()[:, :2] - Ob[:, :2]).max():.1e}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Mkiri = mask_kausal(3) + mask_dari_boolean(
            np.array([False, True, True]))[None, :]
        _, Ak = attn(Qb[1], Kb[1], Vb[1], M=Mkiri)
    print("    pad di kiri + causal: baris 1 =", Ak[0])

    Xd = E[[0, 3]]                       # query decoder: <awal>, cahaya
    Xe = data_mini()                     # memori encoder
    Oc, Ac = attn(Xd @ WQ, Xe @ WK, Xe @ WV)
    tOc = Fn.scaled_dot_product_attention(
        torch.tensor(Xd @ WQ)[None], torch.tensor(Xe @ WK)[None],
        torch.tensor(Xe @ WV)[None])[0]
    print("(3) Cross-attention, 2 query decoder x 3 key encoder:")
    print("    A =", str(Ac).replace("\n", "\n        "))
    print("    O =", str(Oc).replace("\n", "\n        "))
    print(f"    PyTorch: selisih maks = {np.abs(tOc.numpy() - Oc).max():.1e}")

    n = 1024
    print(f"(4) Banyaknya pasangan (query, key) yang dihitung, n = {n}:")
    for nama, kw in (("penuh", {}), ("kausal", {}),
                     ("jendela", dict(w=128)), ("global", dict(w=128))):
        B = pola_mask(nama, n, **kw)
        print(f"    {nama:8s} {B.sum():9,d}  ({B.sum() / n / n:.3f} dari n^2)")
