# Asal data

## data/korpus/ (Bab 13--18)

Teks sembilan peraturan perundang-undangan Republik Indonesia, diambil
dari Wikisumber bahasa Indonesia (<https://id.wikisource.org>) pada
9 Oktober 2026 oleh `data/unduh_korpus.py`. Judul, alamat, dan nomor
revisi setiap halaman tercatat di `data/korpus/sumber.json`, sehingga
korpus dapat diunduh ulang persis.

Menurut Pasal 42 Undang-Undang Nomor 28 Tahun 2014 tentang Hak Cipta,
tidak ada hak cipta atas peraturan perundang-undangan. Setiap halaman
sumber mencantumkan status domain publik itu.

Pembersihan: kotak keterangan, navigasi, nomor halaman, penanda
perubahan UUD, dan lampiran dibuang; spasi dirapikan. Isi pasal tidak
diubah.

## data/model/ (Bab 13--15, 18, Lampiran B)

Titik simpan `GPTMini` hasil `kode/bab13_latih.py` (karakter, bpe,
uud) beserta log pelatihannya (`*.json`) dan aturan BPE
(`bpe1000.json`). Lisensinya sama dengan kode (0BSD).

## Data lain

Data digit 8 x 8 (Bab 16) dimuat dari scikit-learn
(`sklearn.datasets.load_digits`). Data lain dibangkitkan sendiri oleh
kode dengan benih tetap.
