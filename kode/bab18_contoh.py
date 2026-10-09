"""Memeriksa setiap bilangan Contoh Soal Bab 18."""
import math

import numpy as np

# Contoh Soal 18.1: key efektif
assert round(math.exp(1.75), 2) == 5.75 and round(math.exp(2.43), 2) == 11.36
assert round(math.exp(3.88), 1) == 48.4
assert round(np.mean([math.log(t) for t in range(1, 129)]), 2) == 3.88

# Contoh Soal 18.2: ablasi
assert round(math.exp(-0.0016), 3) == 0.998
assert round(math.exp(-3.647), 3) == 0.026 and 1 / 16 == 0.0625
assert round(math.log(16), 3) == 2.773

# Contoh Soal 18.3: salah dengan yakin
assert round(-math.log(0.998), 3) == 0.002
p = 0.002 / 15
assert round(p * 1e4, 2) == 1.33 and round(-math.log(p), 2) == 8.92
assert round(0.002 / 16 + 15 / 16 * 8.92, 2) == 8.36

# Contoh Soal 18.4: selektivitas
assert round(0.859 - 0.520, 3) == 0.339 and round(0.729 - 0.513, 3) == 0.216
assert round(0.859 - 0.275, 3) == 0.584 and 100 * 128 == 12_800
assert round(math.sqrt(0.859 * 0.141 / 12_800), 4) == 0.0031
print("Bab 18: semua Contoh Soal cocok")
