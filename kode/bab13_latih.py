"""Bab 13: melatih GPTMini pada korpus peraturan. Lama (beberapa menit
per konfigurasi), sehingga hasilnya disimpan di data/model/ dan dibaca
kode/bab13_evaluasi.py.
# cek_keluaran: lewati

Jalankan: python3 bab13_latih.py karakter | bpe | uud
"""
import json
import sys
import time

import numpy as np
import torch

from bab10_arsitektur import GPTMini
from bab12_optimisasi import laju_belajar
from bab13_korpus import (MODEL, TokenBPE, TokenKarakter, bagi,
                          muat_korpus)

KONFIG = dict(d=128, h=4, N=4, n=128, B=32, langkah=2000, lr=3e-3,
              pemanasan=100, wd=0.1, benih=20261009)


def siapkan(nama):
    k = muat_korpus()
    latih, val = bagi(k)
    if nama == "uud":
        latih, val = bagi({"uud": k["uud1945"]})
    if nama == "bpe":
        tok = TokenBPE(berkas=MODEL / "bpe1000.json")
    else:
        tok = TokenKarakter(latih + val if nama != "uud" else
                            "".join(muat_korpus().values()))
    return tok, tok.encode(latih), tok.encode(val), len(val)


def batch(ids, B, n, rng):
    i = rng.integers(0, len(ids) - n - 1, size=B)
    return torch.tensor(np.stack([ids[j:j + n + 1] for j in i]))


def loss_val(model, ids, n, B=16, k=20, benih=1):
    rng = np.random.default_rng(benih)
    model.eval()
    with torch.no_grad():
        L = []
        for _ in range(k):
            x = batch(ids, B, n, rng)
            z = model(x[:, :-1])
            L.append(torch.nn.functional.cross_entropy(
                z.reshape(-1, z.shape[-1]), x[:, 1:].reshape(-1)).item())
    model.train()
    return float(np.mean(L))


def latih(nama, c=KONFIG):
    tok, ids_l, ids_v, nkar_v = siapkan(nama)
    torch.manual_seed(c["benih"])
    rng = np.random.default_rng(c["benih"])
    model = GPTMini(V=tok.V, d=c["d"], h=c["h"], N=c["N"], n_maks=c["n"])
    hias = [p for n_, p in model.named_parameters() if p.dim() == 2
            and "tok" not in n_ and "pos" not in n_]
    lain = [p for n_, p in model.named_parameters() if p.dim() != 2
            or "tok" in n_ or "pos" in n_]
    opt = torch.optim.AdamW([dict(params=hias, weight_decay=c["wd"]),
                             dict(params=lain, weight_decay=0.0)],
                            lr=c["lr"], betas=(0.9, 0.95))
    log = dict(langkah=[], latih=[], val=[], konfig=c, V=tok.V,
               parameter=sum(p.numel() for p in model.parameters()),
               token_latih=len(ids_l), token_val=len(ids_v),
               karakter_val=nkar_v)
    t0, buf = time.time(), []
    for t in range(c["langkah"]):
        for g in opt.param_groups:
            g["lr"] = laju_belajar(t, c["lr"], c["pemanasan"], c["langkah"],
                                   minimum=0.1 * c["lr"])
        x = batch(ids_l, c["B"], c["n"], rng)
        z = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            z.reshape(-1, tok.V), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        buf.append(loss.item())
        if (t + 1) % 100 == 0:
            log["langkah"].append(t + 1)
            log["latih"].append(float(np.mean(buf)))
            log["val"].append(loss_val(model, ids_v, c["n"]))
            buf = []
            print(f"{nama} {t + 1:5d} latih {log['latih'][-1]:.4f}"
                  f" val {log['val'][-1]:.4f}", flush=True)
    log["detik"] = time.time() - t0
    MODEL.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL / f"{nama}.pt")
    (MODEL / f"{nama}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    for nama in sys.argv[1:]:
        latih(nama)
