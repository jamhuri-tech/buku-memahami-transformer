# Kode — Memahami Transformer

Repositori pendamping buku **_Memahami Transformer: Dari Attention
sampai Model Bahasa, Matriks demi Matriks_** (Edisi Pertama, 2026) oleh
Mohammad Jamhuri, seri *Memahami*. Tag `edisi-1` menandai kode yang
tepat dipakai untuk mencetak edisi pertama.

Buku ini memakai satu **data mini** dari awal sampai akhir: kalimat
"ilmu itu cahaya" dengan kosakata empat token (`<awal>, ilmu, itu,
cahaya`) dan embedding berdimensi 2. Bobot satu head-nya dipilih
supaya skor attention berupa bilangan bulat kecil, sehingga setiap
langkah lintasan maju dan mundur dapat dihitung tangan. File
`kode/babNN_contoh.py` memeriksa setiap bilangan di kotak Contoh Soal
Bab NN.

**Setiap angka keluaran yang tercetak di buku dihasilkan oleh kode di
sini**, dan `periksa.py` membuktikannya: skrip itu menjalankan ulang
kode setiap bab dan mencocokkan hasilnya dengan blok keluaran yang
tercetak di buku.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)). Asal setiap data dan
dasar hukum pemakaiannya tercatat di `data/SUMBER.md`.

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-transformer.git
cd buku-memahami-transformer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab01_alur.py
```

Setiap skrip dijalankan dari akar repositori. Pembangkit bilangan acak
NumPy dan PyTorch selalu memakai benih tetap (20261009). Semua kode
berjalan di CPU; tidak ada model pralatih yang diunduh.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab (sekitar 40 menit di CPU)
.venv/bin/python periksa.py 04     # Bab 4 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. Bab 1--5, 7, 8, dan 10 sepenuhnya deterministik dan
diperiksa otomatis di GitHub Actions bersama semua Contoh Soal. Bab
lain melatih model kecil dengan PyTorch; pada CPU atau versi pustaka
yang berbeda, angkanya dapat berbeda di digit terakhir, tetapi
kesimpulannya sama.

## Melatih ulang model Bab 13

Titik simpan di `data/model/` dipakai Bab 13, 14, 15, 18, dan
Lampiran B. Untuk melatih ulang (sekitar 10 menit per model di CPU):

```bash
.venv/bin/python kode/bab13_latih.py karakter bpe uud
```

## Struktur

```
kode/
  bab01_data.py           data mini dan satu lintasan maju
  bab10_arsitektur.py     GPTMini (PyTorch) dan tiruan NumPy-nya
  bab13_korpus.py         korpus, tokenizer karakter dan BPE
  babNN_*.py              kode Bab NN
  babNN_contoh.py         pemeriksa hitungan tangan Contoh Soal Bab NN
  lampiran_transformer.py decoder Transformer NumPy satu berkas
data/korpus/              UUD 1945 dan 8 undang-undang (domain publik)
data/unduh_korpus.py      pengunduh korpus dari Wikisumber
data/model/               titik simpan model Bab 13
data/SUMBER.md            asal setiap data dan dasar hukumnya
keluaran/babNN.txt        blok keluaran yang tercetak di Bab NN
gen_gambar.py             pembangkit semua gambar buku
periksa.py                pencocok kode dengan buku
```
