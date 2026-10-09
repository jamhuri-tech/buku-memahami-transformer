"""Memeriksa bahwa kode repositori ini menghasilkan angka di buku.

Untuk setiap keluaran/babNN.txt, semua kode/babNN_*.py dijalankan,
keluarannya digabung, lalu setiap blok yang tercetak di buku harus
muncul utuh (baris-baris berurutan) di dalam gabungan itu. Spasi di
ujung baris diabaikan.

    python periksa.py          # semua bab
    python periksa.py 04 05    # bab tertentu saja
"""
import glob
import re
import subprocess
import sys

pilihan = sys.argv[1:]
masalah = 0
for berkas in sorted(glob.glob("keluaran/bab[0-9][0-9].txt")):
    nn = berkas[-6:-4]
    if pilihan and nn not in pilihan:
        continue
    teks = open(berkas, encoding="utf-8").read()
    blok = re.split(r"^=== blok \d+ ===\n", teks, flags=re.M)[1:]
    keluaran = ""
    for s in sorted(glob.glob(f"kode/bab{nn}_*.py")):
        if "# cek_keluaran: lewati" in open(s, encoding="utf-8").read():
            continue
        hasil = subprocess.run([sys.executable, s], capture_output=True,
                               text=True)
        if hasil.returncode != 0:
            print(f"{s}: GAGAL\n{hasil.stderr[-500:]}")
            masalah += 1
        keluaran += hasil.stdout
    bersih = "\n".join(b.rstrip() for b in keluaran.splitlines())
    cocok = 0
    for b in blok:
        t = "\n".join(x.rstrip() for x in b.strip("\n").splitlines())
        if t in bersih:
            cocok += 1
        else:
            masalah += 1
            print(f"Bab {int(nn)}: blok tidak dihasilkan kode:")
            print("    " + t.replace("\n", "\n    ")[:400])
    print(f"Bab {int(nn)}: {cocok}/{len(blok)} blok cocok")
    if cocok < len(blok):
        print(f"--- keluaran kode Bab {int(nn)} di komputer ini ---")
        print(bersih)
        print("--- akhir keluaran ---")

print("SEMUA COCOK" if masalah == 0 else f"{masalah} masalah")
sys.exit(1 if masalah else 0)
