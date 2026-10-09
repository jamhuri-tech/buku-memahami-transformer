"""Bab 10: encoder, decoder, dan encoder-decoder; model decoder kecil
(GPTMini) dalam PyTorch dan tiruan NumPy-nya; parameter dan FLOP.

Dipakai bab-bab berikutnya:
  GPTMini(V, d, h, N, n_maks)   decoder pre-LN, posisi terlatih,
                                GELU, weight tying (PyTorch)
  maju_numpy(par, ids, h)       lintasan maju yang sama dengan NumPy
  par_numpy(model)              bobot GPTMini sebagai dict NumPy
  flop_per_token(P, N, n, d)    2P + 2Nnd (lintasan maju)
"""
import numpy as np
import torch
import torch.nn as nn

from bab06_multihead import mha
from bab08_ffn import gelu
from bab09_normalisasi import layernorm
from bab01_data import mask_kausal


class Blok(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


class GPTMini(nn.Module):
    def __init__(self, V, d=64, h=4, N=2, n_maks=64):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(n_maks, d)
        self.blok = nn.ModuleList([Blok(d, h) for _ in range(N)])
        self.ln_f = nn.LayerNorm(d)
        self.h = h
        for b in self.blok:                      # inisialisasi berskala
            for w in (b.attn.out_proj.weight, b.ffn[2].weight):
                nn.init.normal_(w, std=0.02 / np.sqrt(2 * N))
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.01)

    def forward(self, ids):
        n = ids.shape[-1]
        x = self.tok(ids) + self.pos(torch.arange(n))
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        for b in self.blok:
            x = b(x, mask)
        return self.ln_f(x) @ self.tok.weight.T   # weight tying


def par_numpy(model):
    """Bobot dalam konvensi buku: X W (bukan X W^T)."""
    f = lambda t: t.detach().double().numpy()
    d = model.tok.weight.shape[1]
    par = dict(E=f(model.tok.weight), P=f(model.pos.weight),
               gf=f(model.ln_f.weight), bf=f(model.ln_f.bias), blok=[])
    for b in model.blok:
        W = f(b.attn.in_proj_weight)
        bb = f(b.attn.in_proj_bias)
        par["blok"].append(dict(
            g1=f(b.ln1.weight), b1=f(b.ln1.bias),
            WQ=W[:d].T, WK=W[d:2 * d].T, WV=W[2 * d:].T,
            bQ=bb[:d], bK=bb[d:2 * d], bV=bb[2 * d:],
            WO=f(b.attn.out_proj.weight).T, bO=f(b.attn.out_proj.bias),
            g2=f(b.ln2.weight), b2=f(b.ln2.bias),
            W1=f(b.ffn[0].weight).T, c1=f(b.ffn[0].bias),
            W2=f(b.ffn[2].weight).T, c2=f(b.ffn[2].bias)))
    return par


def maju_numpy(par, ids, h):
    n = len(ids)
    x = par["E"][ids] + par["P"][:n]
    M = mask_kausal(n)
    for b in par["blok"]:
        a, _ = layernorm(x, b["g1"], b["b1"])
        # bias proyeksi diserap dengan menambah kolom 1 pada masukan
        a1 = np.c_[a, np.ones(n)]
        WQ = np.r_[b["WQ"], b["bQ"][None]]
        WK = np.r_[b["WK"], b["bK"][None]]
        WV = np.r_[b["WV"], b["bV"][None]]
        o, _, _ = mha(a1, WQ, WK, WV, b["WO"], h, M=M)
        x = x + o + b["bO"]
        a, _ = layernorm(x, b["g2"], b["b2"])
        x = x + gelu(a @ b["W1"] + b["c1"]) @ b["W2"] + b["c2"]
    z, _ = layernorm(x, par["gf"], par["bf"])
    return z @ par["E"].T


def blok_param(d, dff, silang=False):
    """Encoder/decoder pre-LN dengan bias; silang = blok decoder seq2seq."""
    attn = 4 * d * d + 4 * d
    ffn = 2 * d * dff + dff + d
    ln = 2 * d
    if silang:
        return 2 * attn + ffn + 3 * ln
    return attn + ffn + 2 * ln


def flop_per_token(P, N, n, d):
    return 2 * P + 2 * N * n * d


if __name__ == "__main__":
    torch.manual_seed(20261009)
    d, dff, h = 512, 2048, 8
    enc = nn.TransformerEncoderLayer(d, h, dff, batch_first=True)
    dec = nn.TransformerDecoderLayer(d, h, dff, batch_first=True)
    hitung = lambda m: sum(p.numel() for p in m.parameters())
    print("(1) Parameter satu blok, d = 512, d_ff = 2048:")
    print(f"    encoder: rumus {blok_param(d, dff):,d},"
          f" PyTorch {hitung(enc):,d}")
    print(f"    decoder seq2seq: rumus {blok_param(d, dff, True):,d},"
          f" PyTorch {hitung(dec):,d}")
    V = 37000
    tot = 6 * blok_param(d, dff) + 6 * blok_param(d, dff, True) + V * d
    print(f"    Transformer-base, 6 + 6 blok, kosakata {V}: {tot:,d}")

    model = GPTMini(V=50, d=32, h=4, N=2, n_maks=16).double()
    ids = torch.randint(50, (10,), generator=torch.Generator()
                        .manual_seed(1))
    with torch.no_grad():
        zt = model(ids[None])[0].numpy()
    zn = maju_numpy(par_numpy(model), ids.numpy(), h=4)
    print("(2) GPTMini (V = 50, d = 32, h = 4, N = 2) lawan NumPy:")
    print(f"    parameter {hitung(model):,d}; selisih maks logit"
          f" = {np.abs(zt - zn).max():.1e}")

    print("(3) FLOP lintasan maju per token, d = 768, N = 12:")
    P = 12 * (12 * 768 ** 2)
    print(f"    parameter non-embedding P = {P:,d}")
    for n in (128, 1024, 8192, 32768):
        f = flop_per_token(P, 12, n, 768)
        print(f"    n = {n:6d}: {f / 1e6:8.1f} MFLOP,"
              f" bagian attention {2 * 12 * n * 768 / f:.3f}")
