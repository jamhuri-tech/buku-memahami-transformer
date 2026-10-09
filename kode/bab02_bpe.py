"""Bab 2: byte-pair encoding (BPE) dari nol.

Fungsi yang dipakai bab-bab berikutnya:
  latih_bpe(frek, k)     k aturan penggabungan dari frekuensi kata
  kodekan(kata, aturan)  memecah satu kata menjadi token
  KORPUS_MINI            empat kata untuk hitungan tangan
  korpus_imbuhan()       kata dasar x imbuhan, frekuensi acak tetap
Tanda akhir kata "_" ditempelkan pada setiap kata sebelum pelatihan.
Bila dua pasangan sama seringnya, yang lebih kecil menurut urutan
leksikografis dipilih, supaya hasilnya dapat diulang dengan tangan.
"""
from collections import Counter

import numpy as np

AKHIR = "_"
KORPUS_MINI = {"makan": 4, "makanan": 2, "minum": 3, "minuman": 1}


def hitung_pasangan(kor):
    """Frekuensi setiap pasangan simbol bertetangga."""
    c = Counter()
    for simbol, f in kor.items():
        for a, b in zip(simbol, simbol[1:]):
            c[(a, b)] += f
    return c


def gabungkan(kor, pas):
    """Mengganti setiap kemunculan pasangan dengan satu simbol baru."""
    baru = {}
    for simbol, f in kor.items():
        hasil, i = [], 0
        while i < len(simbol):
            if i < len(simbol) - 1 and (simbol[i], simbol[i + 1]) == pas:
                hasil.append(simbol[i] + simbol[i + 1])
                i += 2
            else:
                hasil.append(simbol[i])
                i += 1
        baru[tuple(hasil)] = baru.get(tuple(hasil), 0) + f
    return baru


def latih_bpe(frek, k):
    """Mengembalikan daftar (pasangan, frekuensi) dan korpus akhir."""
    kor = {tuple(w) + (AKHIR,): f for w, f in frek.items()}
    aturan = []
    for _ in range(k):
        c = hitung_pasangan(kor)
        if not c:
            break
        terbesar = max(c.values())
        pas = min(p for p in c if c[p] == terbesar)
        aturan.append((pas, terbesar))
        kor = gabungkan(kor, pas)
    return aturan, kor


def kodekan(kata, aturan):
    """Aturan diterapkan berurutan, persis seperti saat pelatihan."""
    simbol = {tuple(kata) + (AKHIR,): 1}
    for pas, _ in aturan:
        simbol = gabungkan(simbol, pas)
    return list(next(iter(simbol)))


DASAR = ["makan", "minum", "baca", "tulis", "ajar", "main", "jalan",
         "pikir", "lihat", "dengar", "kerja", "tanam", "masak",
         "pakai", "kirim", "pilih"]


def korpus_imbuhan(benih=20261009, dasar=DASAR):
    """Kata dasar x imbuhan dengan frekuensi berpola Zipf."""
    awalan = ["", "di"]
    akhiran = ["", "an", "kan", "nya", "lah"]
    rng = np.random.default_rng(benih)
    frek = {}
    for i, d in enumerate(DASAR):
        for a in awalan:
            for b in akhiran:
                if a == "di" and b == "an":
                    continue
                f = int(rng.poisson(200 / (i + 1)) + 1)
                if d in dasar:
                    frek[a + d + b] = f
    return frek


def token_per_kata(frek, aturan):
    kata = sum(frek.values())
    tok = sum(f * len(kodekan(w, aturan)) for w, f in frek.items())
    return tok / kata


if __name__ == "__main__":
    aturan, kor = latih_bpe(KORPUS_MINI, 5)
    print("(1) BPE pada korpus mini:")
    for i, (p, f) in enumerate(aturan, 1):
        print(f"    {i}. ({p[0]}, {p[1]}) -> {p[0] + p[1]},"
              f" frekuensi {f}")
    for s, f in kor.items():
        print(f"    {' '.join(s):22s} x{f}")
    print("    makanlah ->", kodekan("makanlah", aturan))

    frek = korpus_imbuhan()
    huruf = sorted({c for w in frek for c in w})
    aturan, kor = latih_bpe(frek, 60)
    print(f"(2) Korpus imbuhan: {len(frek)} jenis kata,"
          f" {sum(frek.values())} kata, {len(huruf)} huruf")
    print("    12 penggabungan pertama:")
    print("      " + " ".join(p[0] + p[1] for p, _ in aturan[:12]))
    for k in (0, 10, 30, 60):
        print(f"    {k:2d} penggabungan: kosakata {len(huruf) + 1 + k:3d},"
              f" token per kata {token_per_kata(frek, aturan[:k]):.3f}")
    for w in ("dimasakkan", "kirimannya", "membaca"):
        print(f"    {w:11s} ->", " ".join(kodekan(w, aturan)))

    # Kata dasar ganjil dipakai melatih, kata dasar genap tidak.
    latih = korpus_imbuhan(dasar=DASAR[0::2])
    uji = korpus_imbuhan(dasar=DASAR[1::2])
    at, _ = latih_bpe(latih, 60)
    print("    BPE 60 penggabungan dari 8 kata dasar saja:")
    print(f"      token per kata, kata dasar terlihat  "
          f"{token_per_kata(latih, at):.3f}")
    print(f"      token per kata, kata dasar baru      "
          f"{token_per_kata(uji, at):.3f}")
