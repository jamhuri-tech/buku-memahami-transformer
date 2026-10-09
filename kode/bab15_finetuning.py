"""Bab 15: adaptasi model karakter Bab 13 ke tugas klasifikasi: dari
potongan 128 karakter, tebak peraturan asalnya (9 kelas).

Dipakai bab-bab berikutnya:
  LoRA(W, r, alfa)            parametrisasi W + (alfa/r) A B
  pasang_lora(model, r)       LoRA pada semua matriks blok
  Klasifikasi(model, kelas)   GPTMini + rata-rata keluaran + kepala linear
  jalankan(cara, ...)         akurasi validasi dan parameter yang dilatih
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as prm

from bab10_arsitektur import GPTMini
from bab13_korpus import MODEL, TokenKarakter, bagi, muat_korpus
from bab13_latih import KONFIG


class LoRA(nn.Module):
    """Bobot PyTorch berukuran (keluar, masuk): W + (alfa/r) B A."""

    def __init__(self, keluar, masuk, r=4, alfa=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, masuk) / np.sqrt(masuk))
        self.B = nn.Parameter(torch.zeros(keluar, r))
        self.s = alfa / r

    def forward(self, W):
        return W + self.s * self.B @ self.A


def pasang_lora(model, r=4):
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.blok:
        for mod, nama in ((b.attn, "in_proj_weight"),
                          (b.attn.out_proj, "weight"),
                          (b.ffn[0], "weight"), (b.ffn[2], "weight")):
            k, m = getattr(mod, nama).shape
            prm.register_parametrization(mod, nama, LoRA(k, m, r))
    return model


class Klasifikasi(nn.Module):
    def __init__(self, model, kelas):
        super().__init__()
        self.model = model
        d = model.tok.weight.shape[1]
        self.kepala = nn.Linear(d, kelas)

    def wakil(self, ids):
        m = self.model
        n = ids.shape[-1]
        x = m.tok(ids) + m.pos(torch.arange(n))
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        for b in m.blok:
            x = b(x, mask)
        return m.ln_f(x).mean(1)

    def forward(self, ids):
        return self.kepala(self.wakil(ids))


def data_kelas(per_kelas, n=128, benih=20261009):
    k = muat_korpus()
    latih, val = bagi(k)
    tok = TokenKarakter(latih + val)
    rng = np.random.default_rng(benih)
    Xl, yl, Xv, yv = [], [], [], []
    for c, t in enumerate(k.values()):
        tl, tv = bagi({"x": t})
        il, iv = tok.encode(tl), tok.encode(tv)
        for i in rng.integers(0, len(il) - n, per_kelas):
            Xl.append(il[i:i + n])
            yl.append(c)
        for i in rng.integers(0, len(iv) - n, 100):
            Xv.append(iv[i:i + n])
            yv.append(c)
    f = lambda a: torch.tensor(np.array(a))
    return f(Xl), f(yl), f(Xv), f(yv), list(k)


def model_dasar(pralatih=True):
    c = KONFIG
    m = GPTMini(V=79, d=c["d"], h=c["h"], N=c["N"], n_maks=c["n"])
    if pralatih:
        m.load_state_dict(torch.load(MODEL / "karakter.pt"))
    return m


def jalankan(cara, per_kelas=50, langkah=300, lr=None, r=4,
             benih=20261009, tebakan=False):
    torch.manual_seed(benih)
    Xl, yl, Xv, yv, _ = data_kelas(per_kelas)
    m = model_dasar(pralatih=(cara != "nol"))
    if cara == "probe":
        for p in m.parameters():
            p.requires_grad_(False)
    if cara == "lora":
        pasang_lora(m, r)
    k = Klasifikasi(m, 9)
    latih = [p for p in k.parameters() if p.requires_grad]
    lr = lr or {"nol": 1e-3, "probe": 3e-3, "penuh": 3e-4, "lora": 2e-3}[cara]
    opt = torch.optim.AdamW(latih, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(benih)
    for _ in range(langkah):
        i = torch.randint(len(Xl), (32,), generator=g)
        loss = nn.functional.cross_entropy(k(Xl[i]), yl[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    k.eval()
    with torch.no_grad():
        pred = torch.cat([k(Xv[j:j + 100]).argmax(1)
                          for j in range(0, len(Xv), 100)])
    acc = (pred == yv).float().mean().item()
    if tebakan:
        return acc, pred.numpy(), yv.numpy()
    return acc, sum(p.numel() for p in latih)


if __name__ == "__main__":
    Xl, yl, Xv, yv, nama = data_kelas(50)
    print("(1) Tugas: tebak peraturan asal potongan 128 karakter, 9 kelas")
    print(f"    latih {len(Xl)} potongan (50 per kelas)")
    print(f"    validasi {len(Xv)} (100 per kelas, 10% akhir tiap berkas)")
    print("(2) 300 langkah AdamW, batch 32:")
    print("    cara              parameter dilatih   akurasi validasi")
    for cara, label in (("nol", "dari nol"), ("probe", "linear probe"),
                        ("penuh", "fine-tuning penuh"),
                        ("lora", "LoRA r = 4")):
        acc, P = jalankan(cara)
        print(f"    {label:17s} {P:12,d} {acc:15.3f}")
    print("(3) Akurasi validasi menurut banyaknya contoh latih per kelas:")
    print("    per kelas   dari nol   probe   penuh   LoRA")
    for pk in (20, 100, 400):
        a = [jalankan(c, per_kelas=pk)[0]
             for c in ("nol", "probe", "penuh", "lora")]
        print(f"    {pk:9d} " + "".join(f"{v:8.3f}" for v in a))
