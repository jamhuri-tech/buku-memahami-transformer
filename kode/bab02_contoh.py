"""Memeriksa setiap bilangan Contoh Soal Bab 2."""
import numpy as np

from bab01_data import E, SASARAN, data_mini, lintasan_maju, one_hot
from bab02_bpe import (KORPUS_MINI, gabungkan, hitung_pasangan, kodekan,
                       korpus_imbuhan, latih_bpe, token_per_kata, DASAR)
from bab02_embedding import gradien_keluaran, kosinus

r4 = lambda v: np.round(v, 4)

# Contoh Soal 2.1: frekuensi pasangan dan dua penggabungan pertama
kor = {tuple(w) + ("_",): f for w, f in KORPUS_MINI.items()}
c = hitung_pasangan(kor)
harap = {("a", "n"): 9, ("m", "a"): 7, ("n", "_"): 7, ("a", "k"): 6,
         ("k", "a"): 6, ("m", "i"): 4, ("i", "n"): 4, ("n", "u"): 4,
         ("u", "m"): 4, ("m", "_"): 3, ("n", "a"): 2}
assert dict(c) == harap
panjang = lambda k: sum(len(s) * f for s, f in k.items())
assert panjang(kor) == 66 == 24 + 16 + 18 + 8
kor1 = gabungkan(kor, ("a", "n"))
c1 = hitung_pasangan(kor1)
assert c1[("an", "_")] == 7 == 4 + 2 + 1
assert c1[("m", "a")] == c1[("a", "k")] == c1[("k", "an")] == 6
kor2 = gabungkan(kor1, ("an", "_"))
assert panjang(kor2) == 50 == 66 - 9 - 7
aturan, akhir = latih_bpe(KORPUS_MINI, 5)
assert [p for p, _ in aturan] == [("a", "n"), ("an", "_"), ("a", "k"),
                                  ("m", "ak"), ("i", "n")]
assert [f for _, f in aturan] == [9, 7, 6, 6, 4]

# Contoh Soal 2.2: mengodekan makanlah
tok = kodekan("makanlah", aturan)
assert tok == ["mak", "an", "l", "a", "h", "_"]
assert "".join(tok).rstrip("_") == "makanlah"

# teks Subbab 2.3: korpus imbuhan
frek = korpus_imbuhan()
at, _ = latih_bpe(frek, 60)
assert len(frek) == 144 and sum(frek.values()) == 6366
assert [p[0] + p[1] for p, _ in at[:3]] == ["an", "kan", "di"]
assert [p[0] + p[1] for p, _ in at[:5]][-1] == "makan"
latih = korpus_imbuhan(dasar=DASAR[0::2])
uji = korpus_imbuhan(dasar=DASAR[1::2])
a8, _ = latih_bpe(latih, 60)
t1, t2 = token_per_kata(latih, a8), token_per_kata(uji, a8)
assert round(t2, 1) == 6.2 and round(t2 / t1) == 4

# Contoh Soal 2.3: gradien embedding = O^T G_X
O = one_hot([2, 1, 2])
GX = np.array([[1, 0], [2, -1], [-3, 1]])
GE = O.T @ GX
assert np.array_equal(GE, [[0, 0], [2, -1], [-2, 1], [0, 0]])
assert np.allclose(E[2] - 0.5 * GE[2], [2, 0.5])
assert np.array_equal(GE.sum(0), GX.sum(0)) and not GX.sum(0).any()

# Contoh Soal 2.4: bagian keluaran baris cahaya
h = lintasan_maju(data_mini())
P = h["P"]
assert np.allclose(r4(P[:, 3]), [0.0333, 0.6572, 0.4781])
g = (P[:, 3] - np.array([0, 0, 1])) / 3
assert np.allclose(r4(g), [0.0111, 0.2191, -0.1740])
assert np.allclose(r4(h["H2"]), [[1.5, 1.5], [-1, 2], [-0.4320, 3.0040]])
gk = gradien_keluaran(h)
assert np.allclose(r4(gk[3]), [-0.1273, -0.0678])
assert np.isclose(0.0166 - 0.2191 + 0.0752, -0.1273)
assert np.isclose(0.0166 + 0.4382 - 0.5226, -0.0678)
assert np.allclose(r4([0.0111 * 1.5, 0.2191 * 2, -0.1740 * 3.0040,
                       -0.1740 * -0.4320]), [0.0166, 0.4382, -0.5227,
                                              0.0752], atol=1.1e-4)
assert np.allclose(gk.sum(0), 0)
# teks: dua bagian berlawanan arah untuk <awal>, komponen kedua
assert round(gk[0, 1], 2) == 0.09

# Contoh Soal 2.5: kosinus dan logit h'_1
assert np.isclose(kosinus(E[2], E[0]), 1 / np.sqrt(2))
assert np.isclose(kosinus(E[2], E[1]), 1 / np.sqrt(2))
assert np.isclose(kosinus(E[2], E[3]), 0)
h1 = h["H2"][0]
assert round(np.linalg.norm(h1), 4) == 2.1213
assert np.isclose(h1 @ E[2], 3) and np.isclose(h1 @ E[1], 1.5)
assert np.isclose(2.1213 * 1.4142, 3, atol=1e-3)
assert np.isclose(2.1213 * 0.7071, 1.5, atol=1e-3)
# teks: sudut h'_3 terhadap ilmu dan cahaya, logitnya
h3 = h["H2"][2]
sudut = np.degrees(np.arctan2(h3[1], h3[0]))
assert round(sudut - 90) == 8 and round(135 - sudut) == 37
assert round(h3 @ E[3], 2) == 3.44 and round(h3 @ E[1], 2) == 3.00
# teks: norma awal GPT-2
assert round(0.02 * np.sqrt(768), 2) == 0.55
print("Bab 2: semua Contoh Soal cocok")
