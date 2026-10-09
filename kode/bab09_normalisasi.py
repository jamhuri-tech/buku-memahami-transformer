"""Bab 9: LayerNorm dan turunannya, RMSNorm, serta post-LN lawan pre-LN.

Fungsi yang dipakai bab-bab berikutnya:
  layernorm(X, g, b, eps)          normalisasi per baris
  layernorm_mundur(dY, cache)      dX, dg, db
  rmsnorm(X, g, eps)
  latih_salin(pre, lr, ...)        loss akhir tugas salin, post/pre-LN
"""
import numpy as np


def layernorm(X, g=None, b=None, eps=1e-5):
    X = np.asarray(X, dtype=float)
    mu = X.mean(-1, keepdims=True)
    var = X.var(-1, keepdims=True)
    sd = np.sqrt(var + eps)
    Xh = (X - mu) / sd
    g = np.ones(X.shape[-1]) if g is None else g
    b = np.zeros(X.shape[-1]) if b is None else b
    return Xh * g + b, (Xh, sd, g)


def layernorm_mundur(dY, cache):
    Xh, sd, g = cache
    dXh = dY * g
    dX = (dXh - dXh.mean(-1, keepdims=True)
          - Xh * (dXh * Xh).mean(-1, keepdims=True)) / sd
    return dX, (dY * Xh).sum(0), dY.sum(0)


def rmsnorm(X, g=None, eps=1e-6):
    X = np.asarray(X, dtype=float)
    rms = np.sqrt((X * X).mean(-1, keepdims=True) + eps)
    g = np.ones(X.shape[-1]) if g is None else g
    return X / rms * g


def latih_salin(pre, lr, N=12, d=64, h=4, n=32, V=16, langkah=400,
                warmup=0, benih=0):
    """Barisan acak sepanjang n/2 diulang dua kali; model menebak token
    berikutnya. Kembalikan rata-rata loss 50 langkah terakhir."""
    import torch
    torch.manual_seed(benih)
    g = torch.Generator().manual_seed(benih)
    emb = torch.nn.Embedding(V, d)
    pos = torch.nn.Embedding(n, d)
    lap = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(
        d, h, 4 * d, dropout=0.0, batch_first=True, norm_first=pre)
        for _ in range(N)])
    akhir = torch.nn.LayerNorm(d) if pre else torch.nn.Identity()
    kepala = torch.nn.Linear(d, V)
    par = [*emb.parameters(), *pos.parameters(), *lap.parameters(),
           *akhir.parameters(), *kepala.parameters()]
    opt = torch.optim.Adam(par, lr=lr)
    m = torch.nn.Transformer.generate_square_subsequent_mask(n - 1)
    riwayat = []
    for t in range(langkah):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, (t + 1) / warmup) if warmup else lr
        a = torch.randint(V, (32, n // 2), generator=g)
        x = torch.cat([a, a], 1)
        z = emb(x[:, :-1]) + pos(torch.arange(n - 1))
        for l in lap:
            z = l(z, src_mask=m, is_causal=True)
        loss = torch.nn.functional.cross_entropy(
            kepala(akhir(z)).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        riwayat.append(loss.item())
    return float(np.mean(riwayat[-50:])), riwayat


if __name__ == "__main__":
    import torch

    np.set_printoptions(precision=4, suppress=True)
    x = np.array([[0.0, 2, 4], [1, 2, 6]])
    y, cache = layernorm(x, eps=0)
    tY = torch.nn.functional.layer_norm(torch.tensor(x), (3,), eps=0.0)
    print("(1) LayerNorm dan RMSNorm, eps = 0:")
    print("    LN  (0, 2, 4) =", y[0])
    print("    LN  (1, 2, 6) =", y[1])
    print("    RMS (0, 2, 4) =", rmsnorm(x, eps=0)[0])
    print(f"    PyTorch layer_norm: selisih maks ="
          f" {np.abs(tY.numpy() - y).max():.1e}")
    print("    norma baris LN =", np.linalg.norm(y, axis=1),
          " sqrt(3) =", round(np.sqrt(3), 4))

    rng = np.random.default_rng(20261009)
    X = rng.normal(size=(4, 6))
    g, b = rng.normal(size=6), rng.normal(size=6)
    dY = rng.normal(size=(4, 6))
    Y, c = layernorm(X, g, b)
    dX, dg, db = layernorm_mundur(dY, c)
    tX = torch.tensor(X, requires_grad=True)
    tg = torch.tensor(g, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    (torch.nn.functional.layer_norm(tX, (6,), tg, tb) *
     torch.tensor(dY)).sum().backward()
    print("(2) Backward LayerNorm lawan autograd (4 x 6 acak):")
    print(f"    selisih maks dX {np.abs(tX.grad.numpy() - dX).max():.1e},"
          f" dg {np.abs(tg.grad.numpy() - dg).max():.1e},"
          f" db {np.abs(tb.grad.numpy() - db).max():.1e}")
    print(f"    maks |jumlah baris dX| = {np.abs(dX.sum(1)).max():.1e}")
    print(f"    maks |dX . x_hat|      = "
          f"{np.abs((dX * c[0]).sum(1)).max():.1e}")

    print("(3) Post-LN lawan pre-LN, 12 lapisan, 400 langkah Adam:")
    print(f"    batas bawah loss = 15/31 log 16 = {15 / 31 * np.log(16):.4f}")
    print("        lr   post-LN   pre-LN   post-LN + warmup 100")
    for lr in (1e-3, 3e-3):
        a = latih_salin(False, lr)[0]
        b = latih_salin(True, lr)[0]
        c = latih_salin(False, lr, warmup=100)[0]
        print(f"    {lr:6.0e} {a:9.3f} {b:8.3f} {c:12.3f}")
