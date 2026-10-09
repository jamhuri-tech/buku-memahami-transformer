"""Bab 1: satu lintasan maju data mini, dicocokkan dengan PyTorch, dan
banyaknya parameter beberapa konfigurasi Transformer.

Fungsi yang dipakai bab-bab berikutnya:
  banyak_parameter(d, L, dff, V, n_ctx, ...)  parameter satu model
"""
import numpy as np

from bab01_data import (E, KOSAKATA, SASARAN, W1, W2, data_mini,
                        lintasan_maju, loss_ce)


def banyak_parameter(d, L, dff, V, n_ctx, bias=True, tipe=0,
                     ln_emb=False, pooler=False, ln_akhir=True):
    """Parameter Transformer bergaya GPT-2/BERT, keluaran diikat ke E."""
    b = 1 if bias else 0
    attn = 4 * d * d + 4 * d * b            # W_Q, W_K, W_V, W_O
    ffn = 2 * d * dff + (dff + d) * b       # W_1, W_2
    ln = 2 * 2 * d                          # dua LayerNorm (gamma, beta)
    lapisan = attn + ffn + ln
    emb = V * d + n_ctx * d + tipe * d
    total = L * lapisan + emb
    total += 2 * d if ln_akhir else 0
    total += 2 * d if ln_emb else 0
    total += (d * d + d) if pooler else 0
    return dict(attn=attn, ffn=ffn, lapisan=lapisan, emb=emb,
                total=total)


if __name__ == "__main__":
    import torch
    import torch.nn.functional as Fn

    np.set_printoptions(precision=4, suppress=True)
    X = data_mini()
    h = lintasan_maju(X)
    print("(1) Lintasan maju data mini (NumPy):")
    print("    A =", str(h["A"]).replace("\n", "\n        "))
    print("    H' =", str(h["H2"]).replace("\n", "\n         "))
    for t in range(3):
        p = h["P"][t]
        print(f"    t={t + 1} sasaran {KOSAKATA[SASARAN[t]]:7s}"
              f" P = {p}")
    L = loss_ce(h["P"], SASARAN)
    print(f"    loss cross-entropy rata-rata = {L:.4f}")

    # PyTorch: scaled_dot_product_attention dengan is_causal=True
    tX = torch.tensor(X)
    tq, tk, tv = (torch.tensor(h[k]) for k in "QKV")
    o = Fn.scaled_dot_product_attention(tq[None], tk[None], tv[None],
                                        is_causal=True)[0]
    Hm = tX + o
    Hm2 = Hm + torch.relu(Hm @ torch.tensor(W1)) @ torch.tensor(W2)
    logit = Hm2 @ torch.tensor(E).T
    Lt = Fn.cross_entropy(logit, torch.tensor(SASARAN))
    beda = np.abs(o.numpy() - h["O"]).max()
    print(f"    PyTorch: selisih maks O = {beda:.1e},"
          f" loss = {Lt.item():.4f}")

    print("(2) Banyaknya parameter (keluaran diikat ke embedding):")
    konfig = [
        ("data mini", dict(d=2, L=1, dff=4, V=4, n_ctx=0, bias=False,
                           ln_akhir=False)),
        ("GPT-2 kecil", dict(d=768, L=12, dff=3072, V=50257,
                             n_ctx=1024)),
        ("BERT-base", dict(d=768, L=12, dff=3072, V=30522, n_ctx=512,
                           tipe=2, ln_emb=True, pooler=True,
                           ln_akhir=False)),
    ]
    print(f"    {'model':12s} {'per lapisan':>12s} {'embedding':>12s}"
          f" {'total':>13s}")
    for nama, k in konfig:
        r = banyak_parameter(**k)
        if nama == "data mini":
            # tanpa LayerNorm: hanya W_Q, W_K, W_V, W_1, W_2
            r["lapisan"] = 3 * 4 + 8 + 8
            r["total"] = r["lapisan"] + r["emb"]
        print(f"    {nama:12s} {r['lapisan']:12,d} {r['emb']:12,d}"
              f" {r['total']:13,d}")
