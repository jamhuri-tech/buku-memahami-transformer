"""Bab 14: strategi decoding (greedy, suhu, top-k, top-p, beam search),
KV cache dengan NumPy, dan percobaan keragaman teks.

Dipakai bab-bab berikutnya:
  saring(z, suhu, k, p)            logit -> peluang sesudah suhu/top-k/top-p
  beam(f_logp, awal, lebar, pjg)   beam search umum
  DecoderCache(par, h)             lintasan maju bertahap dengan KV cache
"""
import numpy as np

from bab01_data import softmax
from bab08_ffn import gelu
from bab09_normalisasi import layernorm


def saring(z, suhu=1.0, k=None, p=None):
    z = np.asarray(z, dtype=float) / suhu
    if k is not None:
        batas = np.sort(z)[-k]
        z = np.where(z >= batas, z, -np.inf)
    P = softmax(z)
    if p is not None:
        urut = np.argsort(-P)
        kum = np.cumsum(P[urut])
        simpan = urut[: int(np.searchsorted(kum, p) + 1)]
        Q = np.zeros_like(P)
        Q[simpan] = P[simpan]
        P = Q / Q.sum()
    return P


def beam(f_logp, awal, lebar, panjang):
    """f_logp(barisan) -> log-peluang token berikutnya (array)."""
    calon = [(0.0, list(awal))]
    for _ in range(panjang):
        baru = []
        for s, b in calon:
            lp = f_logp(b)
            for v in np.argsort(-lp)[:lebar]:
                baru.append((s + lp[v], b + [int(v)]))
        calon = sorted(baru, key=lambda u: -u[0])[:lebar]
    return calon


class DecoderCache:
    """Memakai bobot par_numpy (Bab 10). Satu token per panggilan."""

    def __init__(self, par, h):
        self.par, self.h = par, h
        self.K = [[] for _ in par["blok"]]
        self.V = [[] for _ in par["blok"]]

    def langkah(self, tok):
        par, h = self.par, self.h
        p = len(self.K[0])                       # posisi token baru
        x = par["E"][tok] + par["P"][p]
        for i, b in enumerate(par["blok"]):
            a, _ = layernorm(x[None], b["g1"], b["b1"])
            q = a[0] @ b["WQ"] + b["bQ"]
            self.K[i].append(a[0] @ b["WK"] + b["bK"])
            self.V[i].append(a[0] @ b["WV"] + b["bV"])
            K, V = np.array(self.K[i]), np.array(self.V[i])
            d = len(q)
            dk = d // h
            o = np.empty(d)
            for j in range(h):
                s = slice(j * dk, (j + 1) * dk)
                w = softmax(K[:, s] @ q[s] / np.sqrt(dk))
                o[s] = w @ V[:, s]
            x = x + o @ b["WO"] + b["bO"]
            a, _ = layernorm(x[None], b["g2"], b["b2"])
            x = x + gelu(a[0] @ b["W1"] + b["c1"]) @ b["W2"] + b["c2"]
        z, _ = layernorm(x[None], par["gf"], par["bf"])
        return z[0] @ par["E"].T



STRATEGI = (("greedy", "greedy"), ("suhu 0.7", dict(suhu=0.7)),
            ("suhu 1.0", dict()), ("suhu 1.3", dict(suhu=1.3)),
            ("top-k 5", dict(k=5)), ("top-p 0.9", dict(p=0.9)))


def siapkan_model():
    from bab13_evaluasi import muat_model
    from bab13_korpus import bagi, muat_korpus, pecah_kata
    import torch as _t
    model, tok, _ = muat_model("karakter")
    latih, val = bagi(muat_korpus())
    kamus = set(w.strip(".,;:()\"") for w in pecah_kata(latih))
    rng = np.random.default_rng(20261009)
    awalan = []
    for _ in range(10):
        i = int(rng.integers(0, len(val) - 200))
        i = val.find(" ", i) + 1
        awalan.append(val[i:i + 40])

    @_t.no_grad()
    def logit_akhir(ids):
        return model(_t.tensor(ids[-128:])[None])[0, -1].double().numpy()

    return tok, kamus, awalan, logit_akhir


def keragaman(panjang=200):
    from bab13_korpus import pecah_kata
    tok, kamus, awalan, logit_akhir = siapkan_model()
    hasil = []
    for nama, kw in STRATEGI:
        r = np.random.default_rng(1)
        teks, lp = [], []
        for a in awalan:
            ids = list(tok.encode(a))
            for _ in range(panjang):
                z = logit_akhir(ids)
                logp = z - np.log(np.exp(z - z.max()).sum()) - z.max()
                P = np.eye(len(z))[np.argmax(z)] if kw == "greedy" \
                    else saring(z, **kw)
                v = int(r.choice(len(P), p=P))
                lp.append(logp[v])
                ids.append(v)
            teks.append(tok.decode(ids[len(a):]))
        kata = [w.strip(".,;:()\"") for t in teks for w in pecah_kata(t)
                if w != "\n"][1:-1]
        nyata = float(np.mean([w in kamus for w in kata if w]))
        g4 = [t[i:i + 4] for t in teks for i in range(len(t) - 3)]
        hasil.append((nama, float(np.mean(lp)), nyata,
                      len(set(g4)) / len(g4)))
    return hasil


def banding_beam():
    tok, _, awalan, logit_akhir = siapkan_model()
    a = list(tok.encode(awalan[0]))
    lpf = lambda b: (lambda z: z - np.log(np.exp(z - z.max()).sum())
                     - z.max())(logit_akhir(b))
    g = beam(lpf, a, 1, 40)[0]
    bb = beam(lpf, a, 4, 40)[0]
    return awalan[0], (g[0], tok.decode(g[1][len(a):])), \
        (bb[0], tok.decode(bb[1][len(a):]))

if __name__ == "__main__":
    import torch

    from bab01_data import data_mini, lintasan_maju
    from bab10_arsitektur import GPTMini, maju_numpy, par_numpy

    np.set_printoptions(precision=4, suppress=True)
    z3 = lintasan_maju(data_mini())["logit"][2]
    print("(1) Posisi 3 data mini:")
    print("    logit    ", z3)
    for nama, kw in (("suhu 1", {}), ("suhu 0.5", dict(suhu=0.5)),
                     ("top-k 2", dict(k=2)), ("top-p 0.7", dict(p=0.7))):
        print(f"    {nama:9s}", saring(z3, **kw))

    torch.manual_seed(20261009)
    model = GPTMini(V=50, d=32, h=4, N=2, n_maks=64).double()
    par = par_numpy(model)
    ids = list(np.random.default_rng(3).integers(0, 50, 40))
    penuh = maju_numpy(par, np.array(ids), h=4)
    dc = DecoderCache(par, h=4)
    bertahap = np.array([dc.langkah(t) for t in ids])
    print("(2) KV cache lawan hitung ulang penuh, 40 token:")
    print(f"    selisih maks logit = {np.abs(penuh - bertahap).max():.1e}")
    print("(3) Pekerjaan untuk menghasilkan n token (per lapisan):")
    print("       n   baris diproyeksikan     hasil kali q.k")
    print("           cache  hitung ulang   cache  hitung ulang")
    for n in (64, 256, 1024):
        t = np.arange(1, n + 1)
        print(f"    {n:4d} {n:6,d} {t.sum():12,d} {t.sum():7,d}"
              f" {(t * (t + 1) // 2).sum():13,d}")


    print("(4) Model karakter, 10 awalan validasi x 200 karakter:")
    print("    strategi      log-p/kar  kata nyata  4-gram unik")
    for nama, lp, nyata, unik in keragaman():
        print(f"    {nama:12s} {lp:9.3f} {nyata:11.3f} {unik:12.3f}")
    aw, g, bb = banding_beam()
    print("(5) Greedy lawan beam search lebar 4, 40 karakter:")
    print(f"    awalan: {aw!r}")
    print(f"    greedy, log-p {g[0]:.3f}:")
    print(f"      {g[1]!r}")
    print(f"    beam 4, log-p {bb[0]:.3f}:")
    print(f"      {bb[1]!r}")
