"""Memeriksa setiap bilangan Contoh Soal Bab 13."""
import json
import math

from bab13_korpus import MODEL

k = json.loads((MODEL / "karakter.json").read_text())
b = json.loads((MODEL / "bpe.json").read_text())

# Contoh Soal 13.1: epoch
T = 2000 * 32 * 128
assert T == 8_192_000
assert k["token_latih"] == 441_077 and b["token_latih"] == 111_034
assert round(T / 441_077, 1) == 18.6 and round(T / 111_034, 1) == 73.8
assert round(73.8 / 18.6, 2) == 3.97

# Contoh Soal 13.2: model NumPy
assert 79 * 16 == 1264 and 3 * 256 == 768 and 2 * 1024 == 2048
assert 1264 + 768 + 2048 == 4080
assert round(math.log(79), 4) == 4.3694

# Contoh Soal 13.3: parameter dan compute
assert 79 * 128 == 10_112 and 128 * 128 == 16_384
assert 12 * 128**2 + 13 * 128 == 198_272 and 4 * 198_272 == 793_088
assert 10_112 + 16_384 + 793_088 + 256 == 819_840 == k["parameter"]
P = 793_088 + 256
assert P == 793_344 and round(6 * P * T / 1e13, 2) == 3.90
assert round(k["detik"] / 60, 1) == 8.4
assert round(3.9e13 / 504 / 1e10, 1) == 7.7
assert b["parameter"] - k["parameter"] == (1080 - 79) * 128

# Contoh Soal 13.4: bit per karakter
assert b["token_val"] == 14_761 and b["karakter_val"] == 50_179
vb = min(b["val"])
assert round(vb, 4) == 3.0335
assert round(3.0335 * 14_761) == 44_777
assert round(44_777 / 50_179, 4) == 0.8923
assert round(vb * 14_761 / 50_179 / math.log(2), 4) == 1.2874
assert round(k["val"][-1], 4) == 0.8421
assert round(k["val"][-1] / math.log(2), 4) == 1.2148
print("Bab 13: semua Contoh Soal cocok")
