"""Bab 13: korpus peraturan perundang-undangan, pembagian latih/validasi,
dan dua tokenizer: karakter dan BPE (Bab 2).

Dipakai bab-bab berikutnya:
  muat_korpus()                 dict nama berkas -> teks
  bagi(teks, frac)              (latih, validasi): 10% akhir tiap berkas
  TokenKarakter(teks)           encode/decode tingkat karakter
  TokenBPE(teks, k)             BPE k penggabungan, disimpan di data/model
"""
import json
import re
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parent.parent
KORPUS = AKAR / "data" / "korpus"
MODEL = AKAR / "data" / "model"


def muat_korpus():
    return {f.stem: f.read_text(encoding="utf-8")
            for f in sorted(KORPUS.glob("*.txt"))}


def bagi(korpus, frac=0.1):
    latih, val = [], []
    for t in korpus.values():
        k = int(len(t) * (1 - frac))
        k = t.rfind("\n", 0, k) + 1          # potong di batas baris
        latih.append(t[:k])
        val.append(t[k:])
    return "".join(latih), "".join(val)


class TokenKarakter:
    def __init__(self, teks):
        self.simbol = sorted(set(teks))
        self.ke_id = {c: i for i, c in enumerate(self.simbol)}

    @property
    def V(self):
        return len(self.simbol)

    def encode(self, teks):
        return np.array([self.ke_id[c] for c in teks], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.simbol[i] for i in ids)


def pecah_kata(teks):
    """Kata = urutan bukan spasi; baris baru menjadi kata tersendiri."""
    return re.findall(r"\n|[^ \n]+", teks)


class TokenBPE:
    """BPE Bab 2 pada kata; tanda akhir kata '_' berarti spasi."""

    def __init__(self, teks=None, k=1000, berkas=None, alfabet=""):
        from bab02_bpe import latih_bpe
        berkas = berkas or MODEL / f"bpe{k}.json"
        if Path(berkas).exists():
            d = json.loads(Path(berkas).read_text(encoding="utf-8"))
            self.aturan = [tuple(p) for p in d["aturan"]]
            self.simbol = d["simbol"]
        else:
            frek = {}
            for w in pecah_kata(teks):
                if w != "\n":
                    frek[w] = frek.get(w, 0) + 1
            at, _ = latih_bpe(frek, k)
            self.aturan = [p for p, _ in at]
            dasar = sorted({c for w in frek for c in w} | set(alfabet)
                           | {"_", "\n"})
            self.simbol = dasar + [a + b for a, b in self.aturan]
            Path(berkas).parent.mkdir(parents=True, exist_ok=True)
            Path(berkas).write_text(json.dumps(dict(
                aturan=self.aturan, simbol=self.simbol), ensure_ascii=False),
                encoding="utf-8")
        self.ke_id = {s: i for i, s in enumerate(self.simbol)}
        self.peringkat = {p: i for i, p in enumerate(self.aturan)}
        self._cache = {}

    @property
    def V(self):
        return len(self.simbol)

    def kata(self, w):
        if w in self._cache:
            return self._cache[w]
        s = list(w) + ["_"]
        while len(s) > 1:
            pas = [(self.peringkat.get((a, b), 10**9), i)
                   for i, (a, b) in enumerate(zip(s, s[1:]))]
            r, i = min(pas)
            if r == 10**9:
                break
            s = s[:i] + [s[i] + s[i + 1]] + s[i + 2:]
        self._cache[w] = s
        return s

    def encode(self, teks):
        ids = []
        for w in pecah_kata(teks):
            if w == "\n":
                ids.append(self.ke_id["\n"])
            else:
                ids.extend(self.ke_id[s] for s in self.kata(w))
        return np.array(ids, dtype=np.int64)

    def decode(self, ids):
        t = "".join(self.simbol[i] for i in ids)
        t = t.replace("_\n", "\n").replace("_", " ")
        return t[:-1] if t.endswith(" ") else t


if __name__ == "__main__":
    k = muat_korpus()
    latih, val = bagi(k)
    print("(1) Korpus: 9 peraturan, Wikisumber, domain publik")
    for nama, t in k.items():
        print(f"    {nama:10s} {len(t):7,d} karakter {len(t.split()):6,d} kata")
    print(f"    latih {len(latih):,d} karakter, validasi {len(val):,d}")
    tc = TokenKarakter(latih + val)
    tb = TokenBPE(latih, 1000, alfabet=val)
    nl, nv = len(tb.encode(latih)), len(tb.encode(val))
    print("(2) Tokenizer:")
    print(f"    karakter: V = {tc.V}, token latih {len(latih):,d}")
    print(f"    BPE 1000: V = {tb.V}, token latih {nl:,d}")
    print(f"    {len(latih) / nl:.2f} karakter per token BPE")
    print("    15 penggabungan pertama:")
    print("      " + " ".join(a + b for a, b in tb.aturan[:15]))
    contoh = "Setiap warga negara berhak mendapat pendidikan."
    print("    kembali utuh:", tb.decode(tb.encode(contoh)) == contoh)
    print("     ", " ".join(tb.simbol[i] for i in tb.encode(contoh)))
