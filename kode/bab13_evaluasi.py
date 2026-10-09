"""Bab 13: membaca hasil kode/bab13_latih.py dari data/model/ --
ringkasan pelatihan, bit per karakter, overfitting, dan contoh teks.

Dipakai bab-bab berikutnya:
  muat_model(nama)                      (model, tokenizer, log)
  hasilkan(model, ids, k, suhu, rng)    sampling sederhana tanpa KV cache
"""
import json

import numpy as np
import torch

from bab10_arsitektur import GPTMini
from bab13_korpus import MODEL
from bab13_latih import KONFIG, siapkan


def muat_model(nama):
    log = json.loads((MODEL / f"{nama}.json").read_text())
    tok = siapkan(nama)[0]
    c = KONFIG
    model = GPTMini(V=tok.V, d=c["d"], h=c["h"], N=c["N"], n_maks=c["n"])
    model.load_state_dict(torch.load(MODEL / f"{nama}.pt"))
    model.eval()
    return model, tok, log


@torch.no_grad()
def hasilkan(model, ids, k, suhu=1.0, rng=None, n_maks=128):
    ids = list(ids)
    for _ in range(k):
        z = model(torch.tensor(ids[-n_maks:])[None])[0, -1].double().numpy()
        if suhu == 0:
            ids.append(int(np.argmax(z)))
            continue
        p = np.exp((z - z.max()) / suhu)
        ids.append(int(rng.choice(len(p), p=p / p.sum())))
    return ids


if __name__ == "__main__":
    ln2 = np.log(2)
    print("(4) Ringkasan pelatihan, 2000 langkah, batch 32 x 128 token:")
    print(f"    {'model':8s} {'V':>4s} {'parameter':>10s} {'latih':>6s}"
          f" {'val':>6s} {'val min':>6s} (langkah)")
    hasil = {}
    for nama in ("karakter", "bpe", "uud"):
        log = json.loads((MODEL / f"{nama}.json").read_text())
        hasil[nama] = log
        j = int(np.argmin(log["val"]))
        print(f"    {nama:8s} {log['V']:4d} {log['parameter']:10,d}"
              f" {log['latih'][-1]:6.3f} {log['val'][-1]:6.3f}"
              f" {log['val'][j]:6.3f} ({log['langkah'][j]})")
    k, b = hasil["karakter"], hasil["bpe"]
    bpk = k["val"][-1] / ln2
    bpb = b["val"][-1] * b["token_val"] / b["karakter_val"] / ln2
    print("(5) Bit per karakter validasi (pembanding yang adil):")
    f = b["token_val"] / b["karakter_val"] / ln2
    print(f"    karakter, akhir    : {bpk:.3f} bit/karakter")
    print(f"    BPE 1000, akhir    : {bpb:.3f} bit/karakter")
    print(f"    BPE 1000, terbaik  : {min(b['val']) * f:.3f} bit/karakter")
    print(f"    (token BPE per karakter validasi"
          f" {b['token_val']:,d}/{b['karakter_val']:,d})")
    print(f"    waktu latih: karakter {k['detik'] / 60:.1f} menit,"
          f" BPE {b['detik'] / 60:.1f} menit")

    model, tok, _ = muat_model("karakter")
    awalan = "Setiap warga negara berhak"
    rng = np.random.default_rng(20261009)
    print(f"(6) Teks model karakter, awalan '{awalan}'")
    for suhu in (0, 0.8):
        ids = hasilkan(model, tok.encode(awalan), 160, suhu, rng)
        teks = tok.decode(ids[len(awalan):]).replace("\n", " / ")
        print(f"    suhu {suhu}:")
        for i in range(0, len(teks), 56):
            print("      " + teks[i:i + 56])
