"""Bab 11: loss model bahasa, perplexity, entropi sebagai batas bawah,
masking loss, masked LM, dan label smoothing.

Fungsi yang dipakai bab-bab berikutnya:
  rantai_markov(V, alpha, benih)     matriks transisi acak
  bangkit_markov(T, B, n, benih)     batch barisan dari rantai itu
  laju_entropi(T)                    H = sum_i pi_i H(T_i)
  ce_halus(Z, y, eps)                cross-entropy dengan label smoothing
  latih_gpt(model, batch, langkah, lr, ...)  loop pelatihan sederhana
"""
import numpy as np
import torch

from bab01_data import softmax


def rantai_markov(V=8, alpha=0.3, benih=20261009):
    rng = np.random.default_rng(benih)
    return rng.dirichlet(np.full(V, alpha), size=V)


def stasioner(T):
    w, v = np.linalg.eig(T.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def laju_entropi(T):
    pi = stasioner(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(T * np.log(T), axis=1)
    return float(pi @ H)


def bangkit_markov(T, B, n, rng):
    V = len(T)
    pi = stasioner(T)
    x = np.empty((B, n), dtype=np.int64)
    x[:, 0] = rng.choice(V, size=B, p=pi)
    C = T.cumsum(1)
    for t in range(1, n):
        u = rng.random(B)
        x[:, t] = (u[:, None] > C[x[:, t - 1]]).sum(1)
    return np.minimum(x, V - 1)


def ce_halus(Z, y, eps):
    """(1 - eps) CE + eps * rata-rata -log p atas kosakata."""
    Z = np.asarray(Z, dtype=float)
    L = np.log(np.exp(Z - Z.max(-1, keepdims=True)).sum(-1)) + Z.max(-1)
    logp = Z - L[..., None]
    nll = -logp[np.arange(len(y)), y]
    seragam = -logp.mean(-1)
    return float(np.mean((1 - eps) * nll + eps * seragam))


def latih_gpt(model, ambil_batch, langkah, lr, pemanasan=0, cetak=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    riwayat = []
    for t in range(langkah):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (t + 1) / pemanasan) if pemanasan else lr
        x = ambil_batch()
        z = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            z.reshape(-1, z.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        riwayat.append(loss.item())
        if cetak and (t + 1) % cetak == 0:
            print(f"    langkah {t + 1:4d}: loss {np.mean(riwayat[-cetak:]):.4f}")
    return riwayat


if __name__ == "__main__":
    from bab01_data import SASARAN, data_mini, lintasan_maju, loss_ce
    from bab10_arsitektur import GPTMini

    P = lintasan_maju(data_mini())["P"]
    L = loss_ce(P, SASARAN)
    print("(1) Data mini:")
    print(f"    loss {L:.4f} nat = {L / np.log(2):.4f} bit per token")
    print(f"    perplexity {np.exp(L):.4f}, seragam {len(P[0])}")

    T = rantai_markov()
    pi = stasioner(T)
    H = laju_entropi(T)
    Hu = float(-(pi * np.log(pi)).sum())
    print("(2) Sumber Markov orde 1, V = 8, transisi Dirichlet(0.3):")
    print(f"    laju entropi H = {H:.4f} (perplexity {np.exp(H):.3f})")
    print(f"    entropi unigram = {Hu:.4f}, log V = {np.log(8):.4f}")
    rng = np.random.default_rng(20261009)
    torch.manual_seed(20261009)
    uji = torch.tensor(bangkit_markov(T, 256, 65, rng))
    model = GPTMini(V=8, d=32, h=4, N=2, n_maks=64)
    ambil = lambda: torch.tensor(bangkit_markov(T, 32, 65, rng))
    latih_gpt(model, ambil, 600, 3e-3)
    with torch.no_grad():
        z = model(uji[:, :-1])
        Lu = torch.nn.functional.cross_entropy(
            z.reshape(-1, 8), uji[:, 1:].reshape(-1)).item()
    # penaksir hitungan bigram dari data latih sebanyak yang dilihat model
    latih = bangkit_markov(T, 600 * 32, 65, np.random.default_rng(1))
    hit = np.ones((8, 8)) * 0.1
    np.add.at(hit, (latih[:, :-1].ravel(), latih[:, 1:].ravel()), 1)
    That = hit / hit.sum(1, keepdims=True)
    u = uji.numpy()
    Lb = -np.mean(np.log(That[u[:, :-1], u[:, 1:]]))
    print(f"    GPTMini sesudah 600 langkah: loss uji {Lu:.4f}")
    print(f"    bigram hitungan (data latih sama): loss uji {Lb:.4f}")

    print("(3) Label smoothing pada logit posisi 2 data mini:")
    Z2 = lintasan_maju(data_mini())["logit"][1:2]
    for eps in (0.0, 0.1, 0.3):
        lt = torch.nn.functional.cross_entropy(
            torch.tensor(Z2), torch.tensor([2]), label_smoothing=eps)
        print(f"    eps = {eps:.1f}: loss {ce_halus(Z2, [2], eps):.4f},"
              f" PyTorch {lt.item():.4f}")
    g = 0.1 / 4
    print(f"    eps = 0.1, V = 4: selisih logit optimum ="
          f" {np.log((1 - 0.1 + g) / g):.4f}")
