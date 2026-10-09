"""Bab 13: pelatihan dengan NumPy saja -- lintasan maju dan mundur Bab 7,
Adam Bab 12 -- dicocokkan langkah demi langkah dengan PyTorch.

Model: satu lapisan, satu head, FFN ReLU, weight tying (seperti data
mini), d = 16, d_ff = 64, pada korpus tingkat karakter.
"""
import numpy as np
import torch

from bab07_turunan import loss, maju_lapisan, mundur_lapisan
from bab12_optimisasi import Adam
from bab13_korpus import TokenKarakter, bagi, muat_korpus

KUNCI = ("E", "WQ", "WK", "WV", "W1", "W2")


def awal(V, d=16, dff=64, benih=20261009):
    rng = np.random.default_rng(benih)
    s = lambda *u: rng.normal(size=u) * 0.1
    return dict(E=s(V, d), WQ=s(d, d), WK=s(d, d), WV=s(d, d),
                W1=s(d, dff), W2=s(dff, d))


def langkah_numpy(par, opt, X):
    g = {k: np.zeros_like(par[k]) for k in KUNCI}
    L = 0.0
    for x in X:
        c = maju_lapisan(par, list(x[:-1]))
        L += loss(c, list(x[1:])) / len(X)
        gi = mundur_lapisan(par, c, list(x[1:]))
        for k in KUNCI:
            g[k] += gi[k] / len(X)
    for k in KUNCI:
        par[k] = opt[k].langkah(par[k], g[k])
    return L


def langkah_torch(t, opt, X):
    Xt = torch.tensor(X)
    E = t["E"]
    L = 0.0
    for x in Xt:
        h = E[x[:-1]]
        Q, K, V = h @ t["WQ"], h @ t["WK"], h @ t["WV"]
        o = torch.nn.functional.scaled_dot_product_attention(
            Q[None], K[None], V[None], is_causal=True)[0]
        H = h + o
        H2 = H + torch.relu(H @ t["W1"]) @ t["W2"]
        L = L + torch.nn.functional.cross_entropy(H2 @ E.T, x[1:]) / len(Xt)
    opt.zero_grad()
    L.backward()
    opt.step()
    return L.item()


if __name__ == "__main__":
    latih, val = bagi(muat_korpus())
    tok = TokenKarakter(latih + val)
    ids = tok.encode(latih)
    par = awal(tok.V)
    t = {k: torch.tensor(v.copy(), requires_grad=True) for k, v in par.items()}
    opt_n = {k: Adam(lr=3e-3) for k in KUNCI}
    opt_t = torch.optim.Adam(list(t.values()), lr=3e-3)
    rng = np.random.default_rng(20261009)
    print("(3) NumPy lawan PyTorch, d = 16, 8 barisan x 32 karakter:")
    beda = 0.0
    for s in range(1, 201):
        i = rng.integers(0, len(ids) - 33, size=8)
        X = np.stack([ids[j:j + 33] for j in i])
        a = langkah_numpy(par, opt_n, X)
        b = langkah_torch(t, opt_t, X)
        beda = max(beda, abs(a - b))
        if s in (1, 50, 100, 200):
            print(f"    langkah {s:3d}: loss NumPy {a:.6f}, PyTorch {b:.6f}")
    bb = max(np.abs(t[k].detach().numpy() - par[k]).max() for k in KUNCI)
    print(f"    selisih maks loss = {beda:.1e}")
    print(f"    selisih maks bobot = {bb:.1e}")
