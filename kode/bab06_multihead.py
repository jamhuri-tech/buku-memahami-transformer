"""Bab 6: multi-head attention dari nol, dicocokkan dengan
torch.nn.MultiheadAttention, dan percobaan beberapa pencarian sekaligus.

Fungsi yang dipakai bab-bab berikutnya:
  mha(X, WQ, WK, WV, WO, h, M=None, Z=None)  keluaran dan bobot per head
  kv_cache(N, h_kv, dk, byte)                 ukuran KV cache per token
  percobaan_cari(h, ...)                      salin beberapa token sekaligus
"""
import numpy as np

from bab01_data import softmax


def pisah(M, h):
    """(n, h*dk) -> (h, n, dk): kolom dibelah menjadi h blok."""
    n, hd = M.shape
    return M.reshape(n, h, hd // h).transpose(1, 0, 2)


def mha(X, WQ, WK, WV, WO, h, M=None, Z=None):
    """Z: memori untuk cross-attention; bila None, self-attention."""
    Z = X if Z is None else Z
    Q, K, V = pisah(X @ WQ, h), pisah(Z @ WK, h), pisah(Z @ WV, h)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])
    if M is not None:
        S = S + M
    A = softmax(S)
    H = A @ V                                   # (h, n, dv)
    gabung = H.transpose(1, 0, 2).reshape(X.shape[0], -1)
    return gabung @ WO, A, H


def kv_cache(N, h_kv, dk, byte=2):
    """Banyaknya bilangan dan byte K dan V per token, N lapisan."""
    bil = 2 * N * h_kv * dk
    return bil, bil * byte


CARI = ("awal", 1, 2, 3)


def percobaan_cari(h, d=16, n=10, V=10, cari=CARI, langkah=1500, benih=0):
    """Satu lapisan attention tanpa FFN harus menyalin beberapa token
    sekaligus: token pertama dan token 1, 2, 3 langkah sebelumnya.
    Mengembalikan akurasi per sasaran dan bobot attention per head."""
    import torch
    F = torch.nn.functional
    torch.manual_seed(benih)
    g = torch.Generator().manual_seed(benih)
    emb = torch.nn.Embedding(V, d)
    pos = torch.nn.Embedding(n, d)
    attn = torch.nn.MultiheadAttention(d, h, batch_first=True, bias=False)
    k = len(cari)
    baca = torch.nn.Linear(d, k * V)
    par = [*emb.parameters(), *pos.parameters(), *attn.parameters(),
           *baca.parameters()]
    opt = torch.optim.Adam(par, lr=3e-3)
    kausal = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)

    def batch(B):
        x = torch.randint(V, (B, n), generator=g)
        ys = []
        for c in cari:
            if c == "awal":
                ys.append(x[:, :1].expand(B, n))
            else:
                ys.append(torch.cat([x[:, :1].expand(B, c), x[:, :-c]], 1))
        return x, torch.stack(ys, -1)

    def maju(x, bobot=False):
        e = emb(x) + pos(torch.arange(n))
        o, A = attn(e, e, e, attn_mask=kausal, need_weights=bobot,
                    average_attn_weights=False)
        return baca(o).view(*x.shape, k, V), A

    for _ in range(langkah):
        x, y = batch(64)
        z, _ = maju(x)
        loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y = batch(2000)
        z, A = maju(x, bobot=True)
        m = max(c for c in cari if c != "awal")
        akurasi = (z[:, m:].argmax(-1) == y[:, m:]).float().mean((0, 1))
    return akurasi.numpy(), A.mean(0).numpy()


if __name__ == "__main__":
    import torch

    from bab01_data import WK, WQ, WV, data_mini, mask_kausal

    np.set_printoptions(precision=4, suppress=True)
    X = data_mini()
    O, A, H = mha(X, WQ, WK, WV, np.eye(2), h=2, M=mask_kausal(3))
    print("(1) Data mini, h = 2, d_k = 1, W_O = I, kausal:")
    for i in range(2):
        print(f"    head {i + 1}: A =",
              str(A[i]).replace("\n", "\n                 "))
    print("    keluaran =", str(O).replace("\n", "\n               "))

    rng = np.random.default_rng(20261009)
    d, h, n = 8, 4, 5
    Xr = rng.normal(size=(n, d))
    W = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]
    Om, _, _ = mha(Xr, *W, h=h, M=mask_kausal(n))
    lap = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True,
                                      dtype=torch.float64)
    with torch.no_grad():
        # PyTorch menyimpan transpos, ditumpuk Q, K, V
        lap.in_proj_weight.copy_(torch.tensor(np.vstack(
            [W[0].T, W[1].T, W[2].T])))
        lap.out_proj.weight.copy_(torch.tensor(W[3].T))
    tX = torch.tensor(Xr)[None]
    tO, _ = lap(tX, tX, tX, attn_mask=torch.tensor(np.isinf(
        mask_kausal(n))))
    print("(2) Acak d = 8, h = 4, kausal, lawan nn.MultiheadAttention:")
    print(f"    selisih maks = {np.abs(tO[0].detach().numpy() - Om).max():.1e}")

    print("(3) Satu lapisan attention, d = 16, salin 4 token sekaligus:")
    print(f"    {'h':>2s}" + "".join(f"{t:>8s}" for t in
                                  ("pertama", "t-1", "t-2", "t-3")))
    for hh in (1, 2, 4):
        a, _ = percobaan_cari(hh)
        print(f"    {hh:2d}" + "".join(f"{v:8.3f}" for v in a))
