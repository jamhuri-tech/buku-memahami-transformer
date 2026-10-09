"""Lampiran B: decoder Transformer pre-LN dengan NumPy saja, memakai
bobot model karakter Bab 13 yang diekspor ke data/model/karakter.npz.
"""
import numpy as np


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    from math import erf, sqrt
    return x * 0.5 * (1 + np.vectorize(erf)(x / sqrt(2)))


def attention(a, W, c, Wo, bo, h):
    """a: n x d sesudah LN; W: d x 3d (kolom Q | K | V)."""
    n, d = a.shape
    q, k, v = np.split(a @ W + c, 3, axis=1)
    pisah = lambda m: m.reshape(n, h, -1).transpose(1, 0, 2)
    q, k, v = pisah(q), pisah(k), pisah(v)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(d // h)
    s = s + np.triu(np.full((n, n), -np.inf), 1)  # mask
    o = softmax(s) @ v                     # h x n x d/h
    o = o.transpose(1, 0, 2).reshape(n, d)
    return o @ Wo + bo


def maju(p, ids, h=4, N=4):
    """Logit n x V untuk barisan id token."""
    x = p["tok"][ids] + p["pos"][:len(ids)]
    for i in range(N):
        g = lambda nama: p[f"{i}.{nama}"]
        a = layernorm(x, g("ln1.g"), g("ln1.b"))
        x = x + attention(a, g("W"), g("c"), g("Wo"), g("bo"),
                          h)
        a = layernorm(x, g("ln2.g"), g("ln2.b"))
        u = gelu(a @ g("W1") + g("c1"))
        x = x + u @ g("W2") + g("c2")
    x = layernorm(x, p["lnf.g"], p["lnf.b"])
    return x @ p["tok"].T                  # weight tying


def hasilkan(p, ids, k, n_maks=128):
    ids = list(ids)
    for _ in range(k):
        ids.append(int(np.argmax(maju(p, ids[-n_maks:])[-1])))
    return ids


def ekspor(berkas_pt, berkas_npz):
    """Mengubah state_dict GPTMini menjadi konvensi X W (bukan X W^T)."""
    import torch
    s = {k: v.double().numpy() for k, v in torch.load(berkas_pt).items()}
    p = {"tok": s["tok.weight"], "pos": s["pos.weight"],
         "lnf.g": s["ln_f.weight"], "lnf.b": s["ln_f.bias"]}
    i = 0
    while f"blok.{i}.ln1.weight" in s:
        b = lambda nama: s[f"blok.{i}.{nama}"]
        p.update({f"{i}.ln1.g": b("ln1.weight"), f"{i}.ln1.b": b("ln1.bias"),
                  f"{i}.W": b("attn.in_proj_weight").T,
                  f"{i}.c": b("attn.in_proj_bias"),
                  f"{i}.Wo": b("attn.out_proj.weight").T,
                  f"{i}.bo": b("attn.out_proj.bias"),
                  f"{i}.ln2.g": b("ln2.weight"), f"{i}.ln2.b": b("ln2.bias"),
                  f"{i}.W1": b("ffn.0.weight").T, f"{i}.c1": b("ffn.0.bias"),
                  f"{i}.W2": b("ffn.2.weight").T, f"{i}.c2": b("ffn.2.bias")})
        i += 1
    np.savez(berkas_npz, **p)


if __name__ == "__main__":
    from pathlib import Path
    import torch
    from bab13_evaluasi import muat_model

    model, tok, _ = muat_model("karakter")
    d = Path(__file__).resolve().parent.parent / "data" / "model"
    ekspor(d / "karakter.pt", d / "karakter.npz")
    p = dict(np.load(d / "karakter.npz"))
    teks = "Pasal 31 (1) Setiap warga negara berhak"
    ids = tok.encode(teks)
    with torch.no_grad():
        zt = model.double()(torch.tensor(ids)[None])[0].numpy()
    print(f"selisih maks logit NumPy - PyTorch: {np.abs(maju(p, ids) - zt).max():.1e}")
    keluar = tok.decode(hasilkan(p, ids, 60)[len(ids):])
    print("greedy:", repr(teks + keluar))
