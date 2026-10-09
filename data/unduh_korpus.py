"""Mengunduh korpus Bab 13 dari Wikisumber bahasa Indonesia.

Isi korpus: UUD 1945 dan delapan undang-undang. Menurut Pasal 42 UU
No. 28 Tahun 2014 tentang Hak Cipta, tidak ada hak cipta atas peraturan
perundang-undangan, dan setiap halaman sumber mencantumkan status domain
publik itu. Teks diambil lewat API MediaWiki (action=parse), lalu
dibersihkan menjadi teks polos. Nomor revisi setiap halaman dicatat di
data/korpus/sumber.json supaya korpus dapat diulang persis.

Jalankan dari akar buku: python3 data/unduh_korpus.py
"""
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HALAMAN = [
    ("uud1945", "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945"),
    ("uu28-2014", "Undang-Undang Republik Indonesia Nomor 28 Tahun 2014"),
    ("uu20-2003", "Undang-Undang Republik Indonesia Nomor 20 Tahun 2003"),
    ("uu12-2012", "Undang-Undang Republik Indonesia Nomor 12 Tahun 2012"),
    ("uu14-2005", "Undang-Undang Republik Indonesia Nomor 14 Tahun 2005"),
    ("uu24-2009", "Undang-Undang Republik Indonesia Nomor 24 Tahun 2009"),
    ("uu11-2008", "Undang-Undang Republik Indonesia Nomor 11 Tahun 2008"),
    ("uu11-2019", "Undang-Undang Republik Indonesia Nomor 11 Tahun 2019"),
    ("uu27-2022", "Undang-Undang Republik Indonesia Nomor 27 Tahun 2022"),
]
API = "https://id.wikisource.org/w/api.php"
AGEN = ("BukuMemahamiTransformer/1.0 "
        "(https://github.com/jamhuri-tech/buku-memahami-transformer)")
KELUAR = Path(__file__).parent / "korpus"


def ambil(judul):
    q = urllib.parse.urlencode(dict(action="parse", page=judul,
                                    prop="text|revid", format="json",
                                    redirects=1))
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": AGEN})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["parse"]


def bersihkan(x):
    """HTML halaman -> teks polos isi peraturan saja."""
    x = re.sub(r"<style.*?</style>", "", x, flags=re.S)
    x = re.sub(r'<span class="pagenum[^"]*".*?</span>\s*</span>', "", x,
               flags=re.S)
    x = re.sub(r"<(br|/p|/li|/tr|/h\d|/div|/dd|/dt)[^>]*>", "\n", x)
    x = re.sub(r"<[^>]+>", "", x)
    t = html.unescape(x).replace("​", "").replace("\xa0", " ")
    # buang kotak keterangan di atas dan navigasi di bawah
    m = re.search(r"Domain publik\s*Domain publik\s*false\s*false", t)
    if m:
        t = t[m.end():]
    t = re.split(r"\n\s*Lihat [Jj]uga\s*\[sunting\]", t)[0]
    t = re.sub(r"\[sunting\]", "", t)
    t = t.replace("\xad", "")
    t = re.sub(r"\s*\u2217+[)/]?", "", t)      # penanda perubahan UUD
    t = re.sub(r"=+\s*(.*?)\s*=+", r"\1", t)    # judul wikitext
    baris = [" ".join(b.split()) for b in t.splitlines()]
    baris = [b for b in baris if b]
    # mulai dari judul peraturan, berhenti sebelum lampiran
    awal = next(i for i, b in enumerate(baris)
                if b.startswith("UNDANG-UNDANG") and b == b.upper())
    baris = baris[awal:]
    if "LAMPIRAN" in baris:
        baris = baris[:baris.index("LAMPIRAN")]
    return "\n".join(baris) + "\n"


if __name__ == "__main__":
    KELUAR.mkdir(exist_ok=True)
    sumber = []
    for nama, judul in HALAMAN:
        p = ambil(judul)
        teks = bersihkan(p["text"]["*"])
        (KELUAR / f"{nama}.txt").write_text(teks, encoding="utf-8")
        url = "https://id.wikisource.org/wiki/" + urllib.parse.quote(
            judul.replace(" ", "_"))
        sumber.append(dict(berkas=f"{nama}.txt", judul=judul, url=url,
                           revisi=p["revid"], karakter=len(teks)))
        print(f"{nama:10s} revisi {p['revid']}  {len(teks):7d} karakter")
        time.sleep(3)
    (KELUAR / "sumber.json").write_text(json.dumps(
        dict(diakses=time.strftime("%Y-%m-%d"), halaman=sumber), indent=1,
        ensure_ascii=False), encoding="utf-8")
    print("jumlah", sum(s["karakter"] for s in sumber), "karakter")
