"""Bab 12: Adam dan AdamW dari nol, jadwal laju belajar, gradient
clipping, dan pengaruh laju belajar pada pelatihan GPTMini.

Fungsi yang dipakai bab-bab berikutnya:
  Adam                          kelas pengoptimal NumPy (AdamW bila wd > 0)
  laju_belajar(t, puncak, ...)  warmup linear + peluruhan kosinus
  potong_norma(gs, c)           gradient clipping norma global
  latih_jadwal(...)             pelatihan GPTMini dengan jadwal dan clipping
"""
import math

import numpy as np
import torch


class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.t, self.m, self.v = 0, None, None

    def langkah(self, theta, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(theta), np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        theta = theta - self.lr * self.wd * theta        # AdamW: terpisah
        return theta - self.lr * mh / (np.sqrt(vh) + self.eps)


def laju_belajar(t, puncak, pemanasan, total, minimum=0.0):
    """t = 0, 1, ...: naik linear selama pemanasan, lalu kosinus."""
    if t < pemanasan:
        return puncak * (t + 1) / pemanasan
    r = (t - pemanasan) / max(1, total - pemanasan)
    return minimum + 0.5 * (puncak - minimum) * (1 + math.cos(math.pi * r))


def potong_norma(gs, c):
    n = math.sqrt(sum(float((g * g).sum()) for g in gs))
    f = min(1.0, c / (n + 1e-12))
    return [g * f for g in gs], n


def latih_jadwal(lr, langkah=300, pemanasan=0, clip=None, opt="adamw",
                 benih=20261009, wd=0.0, kosinus=None):
    from bab10_arsitektur import GPTMini
    from bab11_objektif import bangkit_markov, rantai_markov
    T = rantai_markov()
    rng = np.random.default_rng(benih)
    torch.manual_seed(benih)
    model = GPTMini(V=8, d=32, h=4, N=2, n_maks=64)
    if opt == "sgd":
        o = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        o = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    riwayat, norma = [], []
    for t in range(langkah):
        kos = bool(pemanasan) if kosinus is None else kosinus
        for g in o.param_groups:
            if kos:
                g["lr"] = laju_belajar(t, lr, pemanasan, langkah) \
                    if pemanasan else laju_belajar(t, lr, 0, langkah)
            elif pemanasan:
                g["lr"] = lr * min(1.0, (t + 1) / pemanasan)
            else:
                g["lr"] = lr
        x = torch.tensor(bangkit_markov(T, 32, 65, rng))
        z = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(z.reshape(-1, 8),
                                                 x[:, 1:].reshape(-1))
        o.zero_grad()
        loss.backward()
        n = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip if clip else float("inf"))
        o.step()
        riwayat.append(loss.item() if np.isfinite(loss.item()) else 9.0)
        norma.append(n.item())
    return float(np.mean(riwayat[-30:])), riwayat, norma


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    a = Adam(lr=0.1)
    th = np.array([0.0])
    print("(1) Adam skalar, lr = 0.1, gradien 2 lalu -1:")
    for g in (2.0, -1.0):
        th = a.langkah(th, np.array([g]))
        print(f"    t = {a.t}: m = {a.m[0]:.6f}, v = {a.v[0]:.6f},"
              f" theta = {th[0]:.6f}")
    rng = np.random.default_rng(20261009)
    w = rng.normal(size=5)
    tw = torch.tensor(w.copy(), requires_grad=True)
    ot = torch.optim.AdamW([tw], lr=0.01, weight_decay=0.1)
    an = Adam(lr=0.01, wd=0.1)
    for _ in range(50):
        g = rng.normal(size=5)
        tw.grad = torch.tensor(g)
        ot.step()
        w = an.langkah(w, g)
    print(f"    AdamW 50 langkah lawan PyTorch: selisih maks ="
          f" {np.abs(tw.detach().numpy() - w).max():.1e}")

    print("(2) Laju belajar akhir 30 langkah, GPTMini pada sumber Markov:")
    print("    (H = 1.3568), 300 langkah, batch 32")
    print("        lr      SGD     AdamW   AdamW+warmup")
    for lr in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
        s = latih_jadwal(lr, opt="sgd")[0]
        b = latih_jadwal(lr)[0]
        c = latih_jadwal(lr, pemanasan=30)[0]
        print(f"    {lr:7.0e} {s:8.4f} {b:8.4f} {c:12.4f}")

    print("(3) AdamW lr = 1e-1, empat jadwal:")
    for nama, kw in (("konstan", dict(kosinus=False)),
                     ("warmup 30 saja", dict(pemanasan=30, kosinus=False)),
                     ("kosinus saja", dict(kosinus=True)),
                     ("warmup + kosinus", dict(pemanasan=30))):
        L, r, n = latih_jadwal(1e-1, **kw)
        print(f"    {nama:17s} loss akhir {L:.4f}")
