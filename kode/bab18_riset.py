"""Bab 18: alat analisis -- peta dan entropi attention, induction head
pada tugas salin, ablasi head, dan probing linear per lapisan.

Dipakai di Lampiran:
  telusur(model, ids, matikan)   representasi tiap lapisan dan bobot
                                 attention tiap head; head dapat dimatikan
"""
import numpy as np
import torch

from bab04_attention import entropi


@torch.no_grad()
def telusur(model, ids, matikan=()):
    """matikan: daftar (lapisan, head) yang keluarannya dinolkan."""
    n = ids.shape[-1]
    x = model.tok(ids) + model.pos(torch.arange(n))
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    rep, bobot = [x], []
    for l, b in enumerate(model.blok):
        a = b.ln1(x)
        at = b.attn
        d = a.shape[-1]
        h = at.num_heads
        W, c = at.in_proj_weight, at.in_proj_bias
        q, k, v = (a @ W[i * d:(i + 1) * d].T + c[i * d:(i + 1) * d]
                   for i in range(3))
        sp = lambda t: t.view(*t.shape[:-1], h, d // h).transpose(-2, -3)
        q, k, v = sp(q), sp(k), sp(v)
        s = q @ k.transpose(-1, -2) / np.sqrt(d // h)
        A = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
        o = A @ v
        for (ll, hh) in matikan:
            if ll == l:
                o[..., hh, :, :] = 0
        o = o.transpose(-2, -3).reshape(*a.shape)
        x = x + at.out_proj(o)
        x = x + b.ffn(b.ln2(x))
        rep.append(x)
        bobot.append(A)
    return rep, bobot, model.ln_f(x) @ model.tok.weight.T


def salin_batch(B, n, V, g):
    a = torch.randint(V, (B, n // 2), generator=g)
    return torch.cat([a, a], 1)


def latih_salin(N=2, d=64, h=4, n=32, V=16, langkah=1500, benih=20261009):
    from bab10_arsitektur import GPTMini
    torch.manual_seed(benih)
    g = torch.Generator().manual_seed(benih)
    m = GPTMini(V=V, d=d, h=h, N=N, n_maks=n)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(langkah):
        x = salin_batch(32, n, V, g)
        z = m(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(z.reshape(-1, V),
                                                 x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def skor_head(A, jarak):
    """Rata-rata bobot ke t - 1 dan ke t - jarak, per head."""
    n = A.shape[-1]
    t = torch.arange(1, n)
    prev = A[..., t, t - 1].mean((0, -1))
    t2 = torch.arange(jarak, n)
    lom = A[..., t2, t2 - jarak].mean((0, -1))
    return prev.numpy(), lom.numpy()


def nll_token(logit, x):
    """-log p token sasaran x[:, 1:], untuk logit dari x[:, :-1]."""
    lp = torch.log_softmax(logit, -1)
    return -lp.gather(-1, x[:, 1:, None])[..., 0]


def salin_ulang(B, panjang, n, V, g):
    """Segmen acak sepanjang `panjang` diulang, sisanya acak."""
    x = torch.randint(V, (B, n), generator=g)
    x[:, panjang:2 * panjang] = x[:, :panjang]
    return x


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    from bab13_evaluasi import muat_model
    from bab13_korpus import bagi, muat_korpus
    from bab10_arsitektur import GPTMini

    np.set_printoptions(precision=3, suppress=True)
    model, tok, _ = muat_model("karakter")
    latih, val = bagi(muat_korpus())
    rng = np.random.default_rng(20261009)
    awal = rng.integers(0, len(val) - 129, 64)
    ids = torch.tensor(np.stack([tok.encode(val[i:i + 128]) for i in awal]))
    rep, bobot, _ = telusur(model, ids)
    print("(1) Entropi attention rata-rata (nat), model karakter Bab 13:")
    for l, A in enumerate(bobot):
        H = entropi(A.numpy()).mean((0, 2))
        print(f"    lapisan {l + 1}: " + " ".join(f"{v:5.2f}" for v in H))
    print(f"    seragam atas 128 key: {np.log(128):.2f}")
    print(f"    seragam atas key yang tersedia, rata-rata: "
          f"{np.mean([np.log(t) for t in range(1, 129)]):.2f}")

    n, V = 32, 16
    sm = latih_salin()
    g = torch.Generator().manual_seed(7)
    x = salin_batch(500, n, V, g)
    _, bobot_s, z = telusur(sm, x[:, :-1])
    print("(2) Tugas salin (n = 32, V = 16, jarak salin 16), 2 lapisan:")
    print("    bobot ke t-1 dan ke t-15, head 1..4:")
    for l, A in enumerate(bobot_s):
        pv, lom = skor_head(A, 15)
        print(f"    lapisan {l + 1}: t-1  {pv}")
        print(f"               t-15 {lom}")
    nll = nll_token(z, x)
    L0 = nll[:, 16:].mean().item()
    print(f"    loss paruh kedua {L0:.4f}; paruh pertama"
          f" {nll[:, :15].mean().item():.4f}")
    print(f"    (log 16 = {np.log(16):.4f})")
    for l in range(2):
        r = []
        for hh in range(4):
            nm = nll_token(telusur(sm, x[:, :-1], [(l, hh)])[2], x)
            r.append(nm[:, 16:].mean().item())
        print(f"    matikan satu head lapisan {l + 1}:",
              " ".join(f"{v:.3f}" for v in r))
    print("    uji jarak salin lain (segmen sepanjang k diulang):")
    for k in (16, 12, 8):
        xk = salin_ulang(500, k, n, V, g)
        nk = nll_token(telusur(sm, xk[:, :-1])[2], xk)
        print(f"      k = {k:2d}: loss salinan {nk[:, k:2 * k - 1].mean().item():.4f}")

    print("(3) Probing linear: posisi karakter di dalam kata (0..5+),")
    print("    200 potongan latih, 100 uji; akurasi per lapisan")
    target = []
    def posisi_kata(teks):
        r, k = [], 0
        for ch in teks:
            k = 0 if ch in " \n" else k + 1
            r.append(min(k, 6))
        return r
    awal = rng.integers(0, len(latih) - 129, 300)
    teks = [latih[i:i + 128] for i in awal]
    ids = torch.tensor(np.stack([tok.encode(t) for t in teks]))
    y = np.array([posisi_kata(t) for t in teks])
    torch.manual_seed(0)
    acak = GPTMini(V=tok.V, d=128, h=4, N=4, n_maks=128).eval()
    for nama, m in (("terlatih", model), ("acak", acak)):
        rp, _, _ = telusur(m, ids)
        hasil = []
        for R in rp:
            Xr = R.numpy()
            Xa, ya = Xr[:200].reshape(-1, Xr.shape[-1]), y[:200].ravel()
            Xu, yu = Xr[200:].reshape(-1, Xr.shape[-1]), y[200:].ravel()
            clf = LogisticRegression(max_iter=500).fit(Xa, ya)
            hasil.append(clf.score(Xu, yu))
        print(f"    {nama:8s}: " + " ".join(f"{v:.3f}" for v in hasil))
    print(f"    tebakan kelas terbanyak: {np.mean(y[200:] == np.bincount(y[:200].ravel()).argmax()):.3f}")
