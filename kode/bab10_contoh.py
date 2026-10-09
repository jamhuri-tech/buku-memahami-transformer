"""Memeriksa setiap bilangan Contoh Soal Bab 10."""
import torch

from bab10_arsitektur import GPTMini, blok_param, flop_per_token

# Contoh Soal 10.1: parameter GPTMini
assert 50 * 32 == 1600 and 16 * 32 == 512
assert 4 * 32**2 + 4 * 32 == 4224 and 2 * 32 * 128 + 128 + 32 == 8352
assert 2 * 2 * 32 == 128 and 4224 + 8352 + 128 == 12_704 == blok_param(32, 128)
assert 12 * 32**2 + 13 * 32 == 12_704 and 2 * 12_704 == 25_408
tot = 1600 + 512 + 25_408 + 64
m = GPTMini(V=50, d=32, h=4, N=2, n_maks=16)
assert tot == 27_584 == sum(p.numel() for p in m.parameters())

# Contoh Soal 10.2: Transformer-base
d = 512
assert blok_param(d, 2048) == 12 * d * d + 13 * d == 3_152_384
assert blok_param(d, 2048, True) == 16 * d * d + 19 * d == 4_204_032
assert 12 * d * d == 3_145_728 and 13 * d == 6_656 and 16 * d * d == 4_194_304
assert 6 * 3_152_384 == 18_914_304 and 6 * 4_204_032 == 25_224_192
assert 37_000 * 512 == 18_944_000
assert 18_914_304 + 25_224_192 + 18_944_000 == 63_082_496
assert round((65e6 - 63_082_496) / 512, -2) == 3700
enc = torch.nn.TransformerEncoderLayer(d, 8, 2048)
assert sum(p.numel() for p in enc.parameters()) == 3_152_384

# Contoh Soal 10.3: FLOP GPT-2 kecil
P = 12 * 12 * 768**2
assert P == 84_934_656 and round(2 * P / 1e6, 2) == 169.87
assert round(2 * 12 * 1024 * 768 / 1e6, 2) == 18.87
f = flop_per_token(P, 12, 1024, 768)
assert round(f / 1e6, 1) == 188.7 and round(18.87 / 188.7, 2) == 0.10
assert round(3 * f / 1e6) == 566 and round(6 * P / 1e6) == 510
assert 12 * 768 == 9216
print("Bab 10: semua Contoh Soal cocok")
