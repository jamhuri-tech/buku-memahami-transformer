# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20261009  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().


# ============================ Bab 1 ==================================

def _kotak(ax, x, y, teks, w=2.6, h=0.62, wr=BIRU, isi=BIRU_MUDA, fs=6.3):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=isi, edgecolor=wr, lw=0.7))
    ax.text(x, y, teks, ha="center", va="center", fontsize=fs)


def _panah(ax, a, b, wr=ABU, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7,
                                 color=wr, lw=0.7, linestyle=ls))


def _heat(ax, M, teks, cmap="Blues", vmin=None, vmax=None, mask=None,
          xt=None, yt=None, mask_teks=r"$-\infty$"):
    M = np.asarray(M, dtype=float)
    tampil = np.where(np.isfinite(M), M, np.nan)
    ax.imshow(tampil, cmap=cmap, vmin=vmin, vmax=vmax)
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            if mask is not None and mask[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="#EEEEEE",
                                           edgecolor="white", lw=0.8))
                ax.text(j, i, mask_teks, ha="center", va="center",
                        fontsize=6, color=ABU)
                continue
            v = M[i, j]
            t = teks(v) if callable(teks) else teks[i][j]
            if vmin is not None and vmin < 0:
                gelap = abs(v) > 0.6 * max(-vmin, vmax)
            else:
                gelap = vmax is not None and v > 0.6 * vmax
            ax.text(j, i, t, ha="center", va="center", fontsize=6.3,
                    color="white" if gelap else ABU)
    if xt is not None:
        ax.set_xticks(range(m))
        ax.set_xticklabels(xt, fontsize=6)
    if yt is not None:
        ax.set_yticks(range(n))
        ax.set_yticklabels(yt, fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    kunci_label(ax, "x", "y")


def bab01_alur():
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    lang = [
        (9.0, r"token $w_1, \ldots, w_n$", "id token", JINGGA, JINGGA_MUDA),
        (7.9, r"$\mathbf{X} = $ baris-baris $\mathbf{E}$",
         r"embedding, $n \times d$", BIRU, BIRU_MUDA),
        (6.8, r"$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \mathbf{X}\mathbf{W}_Q,\ \mathbf{X}\mathbf{W}_K,\ \mathbf{X}\mathbf{W}_V$",
         "proyeksi", BIRU, BIRU_MUDA),
        (5.7, r"$\mathbf{A} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k} + \mathbf{M})$",
         r"bobot attention, $n \times n$", HIJAU, HIJAU_MUDA),
        (4.6, r"$\mathbf{H} = \mathbf{X} + \mathbf{A}\mathbf{V}$",
         "residual", HIJAU, HIJAU_MUDA),
        (3.5, r"$\mathbf{H}' = \mathbf{H} + \mathrm{ReLU}(\mathbf{H}\mathbf{W}_1)\mathbf{W}_2$",
         "FFN + residual", HIJAU, HIJAU_MUDA),
        (2.4, r"$\mathbf{P} = \mathrm{softmax}(\mathbf{H}'\mathbf{E}^\top)$",
         r"peluang token, $n \times |\mathcal{V}|$", BIRU, BIRU_MUDA),
        (1.3, r"$L = -\frac{1}{n}\sum_t \log P_{t, y_t}$",
         "loss", MERAH, MERAH_MUDA),
    ]
    for y, rumus, ket, wr, isi in lang:
        _kotak(ax, 4.0, y, rumus, w=5.9, h=0.78, wr=wr, isi=isi)
        ax.text(7.15, y, ket, ha="left", va="center", fontsize=6,
                color=ABU)
    for (y1, *_), (y2, *_) in zip(lang[:-1], lang[1:]):
        _panah(ax, (4.0, y1 - 0.4), (4.0, y2 + 0.4))
    ax.annotate("", xy=(1.05, 4.6), xytext=(1.05, 7.9),
                arrowprops=dict(arrowstyle="-|>", color=ABU, lw=0.6,
                                connectionstyle="arc3,rad=0.3",
                                ls="--"))
    ax.text(0.15, 6.25, "jalur\nresidual", fontsize=5.5, color=ABU,
            ha="left")
    ax.text(4.0, 0.35, r"satu lapisan decoder, $\mathbf{W}_O = \mathbf{I}$, tanpa normalisasi",
            ha="center", fontsize=6, color=ABU)
    simpan(fig, "bab01-alur")


def bab01_attention():
    from bab01_data import KOSAKATA, MASUK, data_mini, lintasan_maju, \
        mask_kausal
    h = lintasan_maju(data_mini())
    tok = [KOSAKATA[i] for i in MASUK]
    mask = np.isinf(mask_kausal(3))
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.1))
    S = h["Q"] @ h["K"].T
    _heat(axs[0], S, lambda v: f"{v:g}", cmap="Blues", vmin=0, vmax=5,
          mask=mask, xt=tok, yt=tok)
    axs[0].set_title(r"skor $\mathbf{Q}\mathbf{K}^\top$ + mask")
    _heat(axs[1], h["A"], lambda v: angka(v, 3), cmap="Greens", vmin=0,
          vmax=1, mask=mask, xt=tok, yt=tok, mask_teks="0")
    axs[1].set_title(r"bobot $\mathbf{A}$ (jumlah baris = 1)")
    for ax in axs:
        ax.set_xlabel("key (token yang dilihat)", fontsize=6)
    axs[0].set_ylabel("query (posisi)", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab01-attention")


def bab01_peluang():
    from bab01_data import KOSAKATA, MASUK, SASARAN, data_mini, \
        lintasan_maju
    P = lintasan_maju(data_mini())["P"]
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.8), sharey=True)
    for t, ax in enumerate(axs):
        warna = [MERAH if v == SASARAN[t] else ABU_GARIS
                 for v in range(4)]
        ax.bar(range(4), P[t], color=warna, width=0.65)
        for v in range(4):
            ax.text(v, P[t, v] + 0.02, angka(P[t, v], 2), ha="center",
                    fontsize=5.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(KOSAKATA, fontsize=5.2, rotation=30)
        kunci_label(ax, "x")
        ax.set_title(f"sesudah “{KOSAKATA[MASUK[t]]}”",
                     fontsize=6.5)
        ax.set_ylim(0, 0.8)
        _rapikan(ax)
    axs[0].set_ylabel("peluang")
    fig.tight_layout()
    simpan(fig, "bab01-peluang")


# ============================ Bab 2 ==================================

def bab02_bpe():
    from bab02_bpe import korpus_imbuhan, latih_bpe, token_per_kata
    frek = korpus_imbuhan()
    aturan, _ = latih_bpe(frek, 120)
    ks = np.arange(0, 121, 5)
    tpk = [token_per_kata(frek, aturan[:k]) for k in ks]
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0))
    axs[0].plot(ks, tpk, color=BIRU, marker="o", ms=2.5, lw=1)
    axs[0].axhline(1, color=ABU_GARIS, ls="--", lw=0.8)
    axs[0].set_xlabel("banyaknya penggabungan")
    axs[0].set_ylabel("token per kata")
    axs[0].set_ylim(0, 9.5)
    fr = [f for _, f in aturan]
    axs[1].scatter(range(1, len(fr) + 1), fr, color=BIRU, s=4)
    for i, (dx, dy) in zip((0, 1, 2, 4), ((8, 0), (8, -6), (8, -12),
                                          (8, -16))):
        p = aturan[i][0]
        axs[1].annotate(p[0] + p[1], (i + 1, fr[i]), fontsize=5.5,
                        xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.3,
                                        color=ABU))
    axs[1].set_yscale("log")
    axs[1].set_xlabel("urutan penggabungan")
    axs[1].set_ylabel("frekuensi pasangan")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-bpe")


def bab02_geometri():
    from bab01_data import E, KOSAKATA, MASUK, data_mini, lintasan_maju
    H2 = lintasan_maju(data_mini())["H2"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for v in range(4):
        ax.annotate("", xy=E[v], xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=BIRU, lw=0.9))
        dx = -0.55 if E[v, 0] < 0 else 0.08
        ax.text(E[v, 0] + dx, E[v, 1] + 0.08, KOSAKATA[v], fontsize=6.5,
                color=BIRU)
    for t in range(3):
        ax.plot(*H2[t], marker="o", ms=4, color=JINGGA)
        ax.text(H2[t, 0] + 0.1, H2[t, 1] - 0.05,
                f"$\\mathbf{{h}}'_{t + 1}$", fontsize=6.5, color=JINGGA)
        ax.plot([0, H2[t, 0]], [0, H2[t, 1]], color=JINGGA, lw=0.5,
                ls=":")
    ax.axhline(0, color=ABU_GARIS, lw=0.6)
    ax.axvline(0, color=ABU_GARIS, lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 2.2)
    ax.set_ylim(-0.4, 3.4)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-geometri")


def bab02_gradien():
    from bab01_data import KOSAKATA
    import bab02_embedding as be
    import torch
    from bab01_data import E, MASUK, SASARAN, W1, W2, WK, WQ, WV, \
        data_mini, lintasan_maju
    h = lintasan_maju(data_mini())
    Gk = be.gradien_keluaran(h)
    t = {k: torch.tensor(v) for k, v in
         dict(WQ=WQ, WK=WK, WV=WV, W1=W1, W2=W2).items()}
    tE = torch.tensor(E, requires_grad=True)
    X = tE[MASUK]
    Q, K, V = X @ t["WQ"], X @ t["WK"], X @ t["WV"]
    O = torch.nn.functional.scaled_dot_product_attention(
        Q[None], K[None], V[None], is_causal=True)[0]
    H = X + O
    H2 = H + torch.relu(H @ t["W1"]) @ t["W2"]
    torch.nn.functional.cross_entropy(
        H2 @ torch.tensor(E).T, torch.tensor(SASARAN)).backward()
    Gm = tE.grad.numpy()
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.9))
    for ax, G, jd in zip(axs, (Gm, Gk, Gm + Gk),
                         ("dari masukan", "dari keluaran", "total")):
        _heat(ax, G, lambda v: angka(v, 2), cmap="RdBu", vmin=-0.9,
              vmax=0.9, xt=["$e_1$", "$e_2$"],
              yt=KOSAKATA if ax is axs[0] else [""] * 4)
        ax.set_title(jd, fontsize=7)
    fig.tight_layout()
    simpan(fig, "bab02-gradien")

# ============================ Bab 3 ==================================

def bab03_sinus():
    from bab03_posisi import pe_sinus
    PE = pe_sinus(64, 32)
    G = pe_sinus(512, 64)
    G = G @ G.T
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.1),
                            gridspec_kw=dict(width_ratios=[1.1, 1]))
    im = axs[0].imshow(PE, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axs[0].set_xlabel("dimensi $j$")
    axs[0].set_ylabel("posisi $p$")
    fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.02)
    k = np.arange(0, 200)
    axs[1].plot(k, G[0, k], color=BIRU, lw=0.9)
    axs[1].set_xlabel("jarak $k$")
    axs[1].set_ylabel(r"$\mathrm{PE}(p)^\top\mathrm{PE}(p+k)$")
    _rapikan(axs[1])
    fig.tight_layout()
    simpan(fig, "bab03-sinus")


def bab03_rope():
    from bab03_posisi import rope, sudut_rope
    rng = np.random.default_rng(BENIH)
    d, n = 32, 32
    q = rng.normal(size=(1, d))
    k = rng.normal(size=(1, d))
    sd = sudut_rope(n, d)
    Qr = rope(np.repeat(q, n, 0), sd)
    Kr = rope(np.repeat(k, n, 0), sd)
    S = Qr @ Kr.T
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.2))
    im = axs[0].imshow(S, cmap="RdBu", vmin=-np.abs(S).max(),
                       vmax=np.abs(S).max())
    axs[0].set_xlabel("posisi key $s$")
    axs[0].set_ylabel("posisi query $p$")
    axs[0].set_title("skor sesudah RoPE", fontsize=7)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
    # sudut putaran data mini: theta = pi/2
    from bab01_data import data_mini, WQ, WK
    Q, K = data_mini() @ WQ, data_mini() @ WK
    sud = np.arange(3)[:, None] * (np.pi / 2)
    Qt, Kt = rope(Q, sud), rope(K, sud)
    ax = axs[1]
    for t in range(3):
        ax.annotate("", xy=Q[t], xytext=(0, 0), arrowprops=dict(
            arrowstyle="-|>", color=ABU_GARIS, lw=0.7))
        ax.annotate("", xy=Qt[t], xytext=(0, 0), arrowprops=dict(
            arrowstyle="-|>", color=BIRU, lw=0.9))
        ax.text(*(Qt[t] * 1.12), f"$\\tilde{{\\mathbf{{q}}}}_{t + 1}$",
                fontsize=6.5, color=BIRU, ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.8, 1.8)
    ax.axhline(0, color=ABU_GARIS, lw=0.5)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    ax.set_title(r"query data mini, $\theta = \pi/2$", fontsize=7)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-rope")


def bab03_alibi():
    H = 8
    m = 2.0 ** (-8 * np.arange(1, H + 1) / H)
    k = np.arange(0, 64)
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    warna = plt.cm.viridis(np.linspace(0, 0.9, H))
    for h in range(H):
        ax.plot(k, -m[h] * k, color=warna[h], lw=0.9,
                label=f"$m = 2^{{-{h + 1}}}$")
    ax.set_xlabel("jarak $t - s$")
    ax.set_ylabel("bias skor")
    ax.set_ylim(-20, 0.5)
    ax.legend(ncol=2, fontsize=5.2, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-alibi")

# ============================ Bab 4 ==================================

def bab04_skala():
    from bab04_attention import simulasi_skala
    from bab01_data import softmax
    rng = np.random.default_rng(BENIH)
    dks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    hasil = np.array([simulasi_skala(dk, ulang=2000, rng=rng) for dk in dks])
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0))
    axs[0].plot(dks, hasil[:, 1], color=MERAH, marker="o", ms=2.5, lw=0.9,
                label="tanpa skala")
    axs[0].plot(dks, hasil[:, 2], color=BIRU, marker="s", ms=2.5, lw=0.9,
                label=r"dibagi $\sqrt{d_k}$")
    axs[0].axhline(1 / 16, color=ABU_GARIS, ls="--", lw=0.8)
    axs[0].set_xscale("log", base=2)
    axs[0].set_xlabel("$d_k$")
    axs[0].set_ylabel("rata-rata bobot terbesar")
    axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize=5.5, loc="center right")
    # satu contoh baris skor untuk d_k = 256
    q = rng.normal(size=256)
    k = rng.normal(size=(16, 256))
    sk = k @ q
    axs[1].bar(np.arange(16) - 0.2, softmax(sk), width=0.4, color=MERAH,
               label="tanpa skala")
    axs[1].bar(np.arange(16) + 0.2, softmax(sk / 16), width=0.4,
               color=BIRU, label="dengan skala")
    axs[1].set_xlabel("key ke-$s$")
    axs[1].set_ylabel("bobot")
    axs[1].legend(fontsize=5.5)
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-skala")


def bab04_suhu():
    from bab01_data import softmax
    c = np.linspace(0, 4, 200)
    S = np.array([1.0, 2.0, 3.0])
    A = np.array([softmax(ci * S) for ci in c])
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    for j, (w, t) in enumerate(zip((ABU, JINGGA, BIRU),
                                   ("<awal>", "ilmu", "itu"))):
        ax.plot(c, A[:, j], color=w, lw=1, label=t)
    for ci, ls in ((1 / np.sqrt(2), ":"), (1, "--")):
        ax.axvline(ci, color=ABU_GARIS, ls=ls, lw=0.8)
    ax.text(1 / np.sqrt(2) - 0.05, 0.92, r"$1/\sqrt{2}$", fontsize=6,
            ha="right")
    ax.text(1.05, 0.92, "$c = 1$", fontsize=6)
    ax.set_xlabel(r"faktor skala $c$ pada skor $(1, 2, 3)$")
    ax.set_ylabel("bobot baris 3")
    ax.legend(fontsize=5.5, loc="center right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-suhu")


def bab04_kernel():
    from bab04_attention import nadaraya_watson
    rng = np.random.default_rng(BENIH)
    xs = np.sort(rng.uniform(0, 6, 40))
    ys = np.sin(xs) + rng.normal(0, 0.3, 40)
    x = np.linspace(0, 6, 300)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0))
    axs[0].scatter(xs, ys, s=4, color=ABU)
    for h, w in ((0.15, JINGGA), (0.6, BIRU)):
        axs[0].plot(x, nadaraya_watson(x, xs, ys, h), color=w, lw=1,
                    label=f"$h = {angka_mat(h, 2)}$")
    axs[0].legend(fontsize=5.5)
    axs[0].set_xlabel("query $x$")
    axs[0].set_ylabel("value $y$")
    xq = np.linspace(0, 6, 60)
    W = np.exp(-(xq[:, None] - xs[None, :]) ** 2 / (2 * 0.6 ** 2))
    W /= W.sum(1, keepdims=True)
    axs[1].imshow(W, cmap="Greens", aspect="auto",
                  extent=[0, 40, 6, 0])
    axs[1].set_xlabel("key ke-$s$ (urut $x_s$)")
    axs[1].set_ylabel("query $x$")
    axs[1].set_title("bobot attention, $h = 0{,}6$", fontsize=7)
    _rapikan(axs[0])
    fig.tight_layout()
    simpan(fig, "bab04-kernel")

# ============================ Bab 5 ==================================

def bab05_pola():
    from bab05_mask import pola_mask
    n = 8
    nama = [("penuh", "penuh (encoder)", {}), ("kausal", "kausal", {}),
            ("jendela", "jendela $w = 3$", {}),
            ("global", "jendela + token global", {}),
            ("awalan", "prefix-LM (awalan 3)", {}),
            ("dokumen", "dua dokumen (3 + 5)", {})]
    fig, axs = plt.subplots(2, 3, figsize=(4.8, 3.3))
    for ax, (k, judul, kw) in zip(axs.flat, nama):
        B = pola_mask(k, n, **kw)
        ax.imshow(B, cmap="Blues", vmin=-0.3, vmax=1.2)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        kunci_label(ax, "x", "y")
        for i in range(n + 1):
            ax.axhline(i - .5, color="white", lw=0.6)
            ax.axvline(i - .5, color="white", lw=0.6)
        ax.set_title(f"{judul}: {B.sum()}", fontsize=6.3)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    simpan(fig, "bab05-pola")


def bab05_batch():
    from bab01_data import mask_kausal
    from bab05_mask import mask_padding
    M = mask_kausal(3)[None] + mask_padding([3, 2], 3)
    tok = [["<awal>", "ilmu", "itu"], ["<awal>", "itu", "<pad>"]]
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 1.9))
    for b, ax in enumerate(axs):
        B = np.isfinite(M[b]).astype(float)
        _heat(ax, B, lambda v: "0" if v else "", cmap="Blues", vmin=-0.3,
              vmax=1.6, mask=~np.isfinite(M[b]), xt=tok[b], yt=tok[b])
        ax.set_title(f"kalimat {b + 1}", fontsize=7)
        ax.set_xlabel("key", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab05-batch")


def bab05_silang():
    from bab01_data import E, WK, WQ, WV, data_mini
    from bab04_attention import attn
    _, A = attn(E[[0, 3]] @ WQ, data_mini() @ WK, data_mini() @ WV)
    fig, ax = plt.subplots(figsize=(3.0, 1.5))
    _heat(ax, A, lambda v: angka(v, 4), cmap="Greens", vmin=0, vmax=0.8,
          xt=["<awal>", "ilmu", "itu"], yt=["<awal>", "cahaya"])
    ax.set_xlabel("key dan value: encoder", fontsize=6)
    ax.set_ylabel("query: decoder", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab05-silang")

# ============================ Bab 6 ==================================

def bab06_mini():
    from bab01_data import KOSAKATA, MASUK, WK, WQ, WV, data_mini, \
        lintasan_maju, mask_kausal
    from bab06_multihead import mha
    X = data_mini()
    _, A, _ = mha(X, WQ, WK, WV, np.eye(2), h=2, M=mask_kausal(3))
    A1 = lintasan_maju(X)["A"]
    tok = [KOSAKATA[i] for i in MASUK]
    mask = np.isinf(mask_kausal(3))
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.8))
    for ax, M, jd in zip(axs, (A1, A[0], A[1]),
                         ("satu head, $d_k = 2$", "head 1, $d_k = 1$",
                          "head 2, $d_k = 1$")):
        _heat(ax, M, lambda v: angka(v, 2), cmap="Greens", vmin=0, vmax=1,
              mask=mask, mask_teks="0", xt=tok,
              yt=tok if ax is axs[0] else [""] * 3)
        ax.set_title(jd, fontsize=6.5)
    fig.tight_layout()
    simpan(fig, "bab06-mini")


def bab06_cari():
    from bab06_multihead import percobaan_cari
    hasil = {h: percobaan_cari(h) for h in (1, 2, 4)}
    fig, axs = plt.subplots(1, 5, figsize=(4.8, 1.75),
                            gridspec_kw=dict(width_ratios=[1.6, 1, 1, 1, 1]))
    sasaran = ["pertama", "$t-1$", "$t-2$", "$t-3$"]
    for j, (h, w) in enumerate(zip((1, 2, 4), (MERAH, JINGGA, BIRU))):
        axs[0].bar(np.arange(4) + (j - 1) * 0.27, hasil[h][0], width=0.27,
                   color=w, label=f"$h = {h}$")
    axs[0].set_xticks(range(4))
    axs[0].set_xticklabels(sasaran, fontsize=5.2, rotation=30)
    kunci_label(axs[0], "x")
    axs[0].set_ylim(0, 1.05)
    axs[0].set_ylabel("akurasi")
    axs[0].legend(fontsize=5, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, 1.0), handlelength=1,
                  columnspacing=0.6)
    _rapikan(axs[0])
    A = hasil[4][1]
    for i in range(4):
        ax = axs[i + 1]
        ax.imshow(A[i], cmap="Greens", vmin=0, vmax=1)
        ax.set_title(f"head {i + 1}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    simpan(fig, "bab06-cari")


def bab06_kv():
    from bab06_multihead import kv_cache
    N, dk, h = 32, 128, 32
    g = np.array([32, 16, 8, 4, 2, 1])
    mb = np.array([kv_cache(N, k, dk)[1] for k in g]) * 4096 / 2**20
    fig, ax = plt.subplots(figsize=(4.2, 1.8))
    ax.bar(range(len(g)), mb, color=[BIRU] + [HIJAU] * 4 + [JINGGA])
    for i, v in enumerate(mb):
        ax.text(i, v + 40, f"{v:.0f}", ha="center", fontsize=5.8)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{k}" for k in g])
    kunci_label(ax, "x")
    ax.set_xlabel("banyaknya head key/value $h_{kv}$ (MHA = 32, MQA = 1)")
    ax.set_ylabel("KV cache 4096 token (MB)")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-kv")

# ============================ Bab 7 ==================================

def bab07_alir():
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    simpul = {"Q": (0.9, 4.2), "K": (0.9, 2.7), "V": (0.9, 1.0),
              "S": (3.6, 3.45), "A": (5.9, 3.45), "O": (8.7, 2.2)}
    teks = {"Q": r"$\mathbf{Q}$", "K": r"$\mathbf{K}$", "V": r"$\mathbf{V}$",
            "S": r"$\mathbf{S}$", "A": r"$\mathbf{A}$", "O": r"$\mathbf{O}$"}
    for k, (x, y) in simpul.items():
        _kotak(ax, x, y, teks[k], w=0.9, h=0.6,
               wr=MERAH if k == "O" else BIRU,
               isi=MERAH_MUDA if k == "O" else BIRU_MUDA, fs=7)
    for a, b in (("Q", "S"), ("K", "S"), ("S", "A"), ("A", "O"),
                 ("V", "O")):
        (x1, y1), (x2, y2) = simpul[a], simpul[b]
        _panah(ax, (x1 + 0.47, y1), (x2 - 0.47, y2))
    lbl = [
        (2.4, 4.45, r"$d\mathbf{Q} = d\mathbf{S}\,\mathbf{K}/\sqrt{d_k}$"),
        (2.4, 2.45, r"$d\mathbf{K} = d\mathbf{S}^\top\mathbf{Q}/\sqrt{d_k}$"),
        (4.75, 3.9, r"$d\mathbf{S} = \mathbf{A}\odot(d\mathbf{A} - \mathrm{rowsum})$"),
        (7.6, 3.4, r"$d\mathbf{A} = d\mathbf{O}\,\mathbf{V}^\top$"),
        (5.0, 1.15, r"$d\mathbf{V} = \mathbf{A}^\top d\mathbf{O}$"),
    ]
    for x, y, t in lbl:
        ax.text(x, y, t, fontsize=6, color=MERAH, ha="center")
    ax.text(8.7, 1.45, r"$d\mathbf{O}$ dari atas", fontsize=6,
            color=MERAH, ha="center")
    ax.text(5.0, 0.2, "panah: arah maju; rumus merah: arah mundur",
            fontsize=6, color=ABU, ha="center")
    simpan(fig, "bab07-alir")


def bab07_jacobian():
    from bab01_data import data_mini, lintasan_maju
    from bab07_turunan import jacobian_softmax
    a3 = lintasan_maju(data_mini())["A"][2]
    J = jacobian_softmax(a3)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0),
                            gridspec_kw=dict(width_ratios=[1, 1.3]))
    _heat(axs[0], J, lambda v: angka(v, 4), cmap="RdBu", vmin=-0.3,
          vmax=0.3, xt=["1", "2", "3"], yt=["1", "2", "3"])
    axs[0].set_title(r"$\mathbf{J}$ baris 3 data mini", fontsize=7)
    p = np.linspace(0, 1, 200)
    axs[1].plot(p, 2 * p * (1 - p), color=BIRU, lw=1)
    axs[1].axvline(0.5, color=ABU_GARIS, ls="--", lw=0.8)
    axs[1].set_xlabel("bobot $p$ (dua key)")
    axs[1].set_ylabel(r"$\|\mathbf{J}\|_F = 2p(1-p)$")
    _rapikan(axs[1])
    fig.tight_layout()
    simpan(fig, "bab07-jacobian")


def bab07_saturasi():
    from bab01_data import softmax
    from bab07_turunan import jacobian_softmax
    rng = np.random.default_rng(BENIH)
    dks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    nt, nd = [], []
    for dk in dks:
        a, b = [], []
        for _ in range(800):
            s = rng.normal(size=(16, dk)) @ rng.normal(size=dk)
            a.append(np.linalg.norm(jacobian_softmax(softmax(s))))
            b.append(np.linalg.norm(jacobian_softmax(softmax(s / np.sqrt(dk)))))
        nt.append(np.mean(a))
        nd.append(np.mean(b))
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.plot(dks, nt, color=MERAH, marker="o", ms=2.5, lw=0.9,
            label="tanpa skala")
    ax.plot(dks, nd, color=BIRU, marker="s", ms=2.5, lw=0.9,
            label=r"dibagi $\sqrt{d_k}$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$d_k$")
    ax.set_ylabel(r"rata-rata $\|\mathbf{J}\|_F$")
    ax.set_ylim(0, 0.4)
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-saturasi")

# ============================ Bab 8 ==================================

def bab08_aktivasi():
    from bab08_ffn import dgelu, dsilu, gelu, relu, silu
    x = np.linspace(-3, 3, 400)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0))
    for f, w, t in ((relu, ABU, "ReLU"), (gelu, BIRU, "GELU"),
                    (silu, JINGGA, "SiLU")):
        axs[0].plot(x, f(x), color=w, lw=1, label=t)
    axs[1].plot(x, (x > 0).astype(float), color=ABU, lw=1)
    axs[1].plot(x, dgelu(x), color=BIRU, lw=1)
    axs[1].plot(x, dsilu(x), color=JINGGA, lw=1)
    axs[0].set_title("fungsi", fontsize=7)
    axs[1].set_title("turunan", fontsize=7)
    axs[0].legend(fontsize=5.5)
    for ax in axs:
        ax.axhline(0, color=ABU_GARIS, lw=0.5)
        ax.axvline(0, color=ABU_GARIS, lw=0.5)
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-aktivasi")


def bab08_residual():
    from bab08_ffn import norma_gradien
    Ns = [1, 2, 4, 8, 16, 32, 64]
    seri = [(dict(residual=False), MERAH, r"tanpa residual, $U \times 1$"),
            (dict(residual=False, skala=0.5), JINGGA,
             r"tanpa residual, $U \times 0{,}5$"),
            (dict(residual=True), ABU, r"residual, $U \times 1$"),
            ("skala", BIRU, r"residual, $U/\sqrt{2N}$")]
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    for kw, w, t in seri:
        if kw == "skala":
            v = [norma_gradien(N, True, skala=1 / np.sqrt(2 * N)) for N in Ns]
        else:
            v = [norma_gradien(N, **kw) for N in Ns]
        ax.plot(Ns, v, color=w, marker="o", ms=2.5, lw=0.9, label=t)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.axhline(1, color=ABU_GARIS, ls="--", lw=0.7)
    ax.set_xlabel("banyaknya lapisan $N$")
    ax.set_ylabel(r"$\|d\mathbf{x}_0\| / \|d\mathbf{x}_N\|$")
    ax.legend(fontsize=5.2, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-residual")


def bab08_stream():
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.plot([0.4, 9.6], [2.3, 2.3], color=BIRU, lw=2.2)
    ax.text(0.4, 2.55, r"$\mathbf{x}_0$", fontsize=7, color=BIRU)
    ax.text(9.2, 2.55, r"$\mathbf{x}_N$", fontsize=7, color=BIRU)
    xs = [1.7, 3.6, 5.5, 7.4]
    for i, x0 in enumerate(xs):
        nama = "attention" if i % 2 == 0 else "FFN"
        y = 3.75 if i % 2 == 0 else 0.85
        _kotak(ax, x0 + 0.45, y, nama, w=1.5, h=0.6,
               wr=HIJAU, isi=HIJAU_MUDA, fs=6)
        _panah(ax, (x0, 2.35 if y > 2.3 else 2.25), (x0, y - 0.32 if y > 2.3
                                                        else y + 0.32),
               wr=ABU)
        _panah(ax, (x0 + 0.9, y - 0.32 if y > 2.3 else y + 0.32),
               (x0 + 0.9, 2.35 if y > 2.3 else 2.25), wr=HIJAU)
        ax.text(x0 - 0.1, (2.3 + y) / 2, "baca", fontsize=5.3, color=ABU,
                ha="right", va="center")
        ax.text(x0 + 1.0, (2.3 + y) / 2, "tulis (+)", fontsize=5.3,
                color=HIJAU, ha="left", va="center")
    ax.text(5.0, 0.05, r"$\mathbf{x}_N = \mathbf{x}_0 + \sum_\ell F_\ell(\mathbf{x}_{\ell-1})$",
            fontsize=7, ha="center", color=ABU)
    simpan(fig, "bab08-stream")

# ============================ Bab 9 ==================================

def bab09_geometri():
    from bab09_normalisasi import layernorm, rmsnorm
    rng = np.random.default_rng(BENIH)
    X = rng.normal(size=(300, 3)) * [1.0, 2, 0.5] + [1.0, 0, 2]
    Y, _ = layernorm(X, eps=0)
    R = rmsnorm(X, eps=0)
    # basis ortonormal bidang tegak lurus (1, 1, 1)
    u = np.array([1, -1, 0]) / np.sqrt(2)
    v = np.array([1, 1, -2]) / np.sqrt(6)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.3))
    axs[0].scatter(X @ u, X @ v, s=3, color=ABU_GARIS, label="masukan")
    axs[0].scatter(Y @ u, Y @ v, s=3, color=BIRU, label="sesudah LN")
    t = np.linspace(0, 2 * np.pi, 200)
    axs[0].plot(np.sqrt(3) * np.cos(t), np.sqrt(3) * np.sin(t),
                color=BIRU, lw=0.5, ls="--")
    axs[0].set_title(r"proyeksi ke bidang $\perp (1,1,1)$", fontsize=6.5)
    axs[0].legend(fontsize=5, loc="lower left", markerscale=2)
    axs[0].set_aspect("equal")
    nX = np.linalg.norm(X, axis=1)
    axs[1].scatter(nX, np.linalg.norm(R, axis=1), s=6, color=JINGGA,
                   label="RMSNorm")
    axs[1].scatter(nX, np.linalg.norm(Y, axis=1), s=2, color=BIRU,
                   label="LayerNorm")
    axs[1].set_xlabel("norma baris masukan")
    axs[1].set_ylabel("norma baris keluaran")
    axs[1].set_ylim(0, 3)
    axs[1].set_title(r"keduanya tepat $\sqrt{3}$", fontsize=6.5)
    axs[1].legend(fontsize=5)
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-geometri")


def bab09_prepost():
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.6))
    for ax, judul, pre in zip(axs, ("post-LN", "pre-LN"), (False, True)):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(judul, fontsize=7.5)
        ax.plot([1.0, 1.0], [0.3, 5.7], color=BIRU, lw=1.8)
        ax.text(1.0, 0.05, r"$\mathbf{x}_\ell$", fontsize=6.5,
                ha="center", color=BIRU)
        ax.text(1.0, 5.75, r"$\mathbf{x}_{\ell+1}$", fontsize=6.5,
                ha="center", color=BIRU)
        if pre:
            blok = [(1.6, "LN"), (2.6, "attention")]
            blok2 = [(3.9, "LN"), (4.9, "FFN")]
            for (y, t) in blok + blok2:
                _kotak(ax, 3.0, y, t, w=1.7, h=0.6,
                       wr=MERAH if t == "LN" else HIJAU,
                       isi=MERAH_MUDA if t == "LN" else HIJAU_MUDA, fs=6)
            for y0, y1, y2 in ((1.0, 1.6, 2.6), (3.3, 3.9, 4.9)):
                _panah(ax, (1.0, y0), (2.15, y1))
                _panah(ax, (3.0, y1 + 0.3), (3.0, y2 - 0.3))
                _panah(ax, (2.15, y2), (1.05, y2 + 0.35), wr=HIJAU)
                ax.text(1.15, y2 + 0.45, "+", fontsize=8, color=HIJAU)
            ax.text(2.6, 0.4, "jalur identitas utuh", fontsize=5.5,
                    color=ABU)
        else:
            for (y, t) in ((1.6, "attention"), (3.9, "FFN")):
                _kotak(ax, 3.0, y, t, w=1.7, h=0.6, wr=HIJAU,
                       isi=HIJAU_MUDA, fs=6)
            for (y, t) in ((2.85, "LN"), (5.15, "LN")):
                _kotak(ax, 1.0, y, t, w=0.9, h=0.5, wr=MERAH,
                       isi=MERAH_MUDA, fs=6)
            for y0, y1, yln in ((0.9, 1.6, 2.85), (3.3, 3.9, 5.15)):
                _panah(ax, (1.0, y0), (2.15, y1))
                _panah(ax, (3.0, y1 + 0.3), (1.5, yln - 0.1), wr=HIJAU)
            ax.text(2.6, 0.4, "LN memotong jalur identitas", fontsize=5.5,
                    color=ABU)
    fig.tight_layout()
    simpan(fig, "bab09-prepost")


def bab09_kurva():
    from bab09_normalisasi import latih_salin
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    for (pre, wu), w, t in (((False, 0), MERAH, "post-LN"),
                            ((True, 0), BIRU, "pre-LN"),
                            ((False, 100), JINGGA, "post-LN + warmup")):
        _, r = latih_salin(pre, 3e-3, warmup=wu)
        r = np.convolve(r, np.ones(10) / 10, mode="valid")
        ax.plot(np.arange(len(r)) + 10, r, color=w, lw=0.9, label=t)
    ax.axhline(15 / 31 * np.log(16), color=ABU_GARIS, ls="--", lw=0.8)
    ax.axhline(np.log(16), color=ABU_GARIS, ls=":", lw=0.8)
    ax.text(395, np.log(16) + 0.05, r"$\log 16$", fontsize=5.5,
            ha="right")
    ax.text(300, 15 / 31 * np.log(16) + 0.04, "batas bawah", fontsize=5.5,
            ha="right")
    ax.set_xlabel("langkah")
    ax.set_ylabel("loss (rata-rata bergerak 10)")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-kurva")

# ============================ Bab 10 =================================

def bab10_tiga():
    from bab05_mask import pola_mask
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 2.0))
    n = 6
    judul = ["encoder (BERT)", "decoder (GPT)", "encoder-decoder (T5)"]
    for ax, j in zip(axs, judul):
        ax.set_title(j, fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(pola_mask("penuh", n), cmap="Blues", vmin=-0.3, vmax=1.2)
    axs[1].imshow(pola_mask("kausal", n), cmap="Blues", vmin=-0.3, vmax=1.2)
    # gabungan: 3 token sumber (penuh), 3 token sasaran (kausal + silang)
    B = np.zeros((n, n))
    B[:3, :3] = 1                       # encoder penuh
    B[3:, :3] = 0.6                     # cross-attention
    B[3:, 3:] = np.tril(np.ones((3, 3)))
    axs[2].imshow(B, cmap="Blues", vmin=-0.3, vmax=1.2)
    axs[2].axhline(2.5, color="white", lw=1.5)
    axs[2].axvline(2.5, color="white", lw=1.5)
    axs[2].text(1, 4, "silang", fontsize=5.5, ha="center", va="center",
                color="white")
    for ax in axs:
        for i in range(n + 1):
            ax.axhline(i - .5, color="white", lw=0.4)
            ax.axvline(i - .5, color="white", lw=0.4)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    simpan(fig, "bab10-tiga")


def bab10_param():
    from bab10_arsitektur import blok_param
    model = [("GPT-2 kecil", 768, 12, 50257, 1024, False),
             ("BERT-base", 768, 12, 30522, 512, False),
             ("Transformer-base", 512, 6, 37000, 0, True)]
    fig, ax = plt.subplots(figsize=(4.4, 1.8))
    for i, (nm, d, N, V, nm_, seq) in enumerate(model):
        emb = V * d + nm_ * d
        attn = (4 * d * d + 4 * d) * N * (3 if seq else 1)
        ffn = (8 * d * d + 5 * d) * N * (2 if seq else 1)
        kiri = 0
        for v, w, t in ((emb, ABU_GARIS, "embedding"),
                        (attn, BIRU, "attention"), (ffn, HIJAU, "FFN")):
            ax.barh(i, v / 1e6, left=kiri, color=w,
                    label=t if i == 0 else None)
            kiri += v / 1e6
        ax.text(kiri + 1, i, f"{kiri:.0f} juta", va="center", fontsize=5.8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([m[0] for m in model], fontsize=6)
    kunci_label(ax, "y")
    ax.set_xlabel("parameter (juta)")
    ax.set_xlim(0, 150)
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-param")


def bab10_flop():
    from bab10_arsitektur import flop_per_token
    P = 12 * 12 * 768 ** 2
    n = np.logspace(1, 5, 200)
    f = flop_per_token(P, 12, n, 768)
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    ax.plot(n, np.full_like(n, 2 * P) / 1e6, color=BIRU, lw=1,
            label="parameter, $2P$")
    ax.plot(n, 2 * 12 * n * 768 / 1e6, color=JINGGA, lw=1,
            label="attention, $2Nnd$")
    ax.plot(n, f / 1e6, color=ABU, lw=1, ls="--", label="total")
    ax.axvline(12 * 768, color=ABU_GARIS, lw=0.7, ls=":")
    ax.text(12 * 768 * 1.1, 30, "$n = 12d$", fontsize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("panjang konteks $n$")
    ax.set_ylabel("MFLOP per token")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-flop")


# ============================ Bab 11 =================================

def bab11_markov():
    import torch
    from bab10_arsitektur import GPTMini
    from bab11_objektif import (bangkit_markov, laju_entropi, latih_gpt,
                                rantai_markov, stasioner)
    T = rantai_markov()
    pi = stasioner(T)
    rng = np.random.default_rng(BENIH)
    torch.manual_seed(BENIH)
    model = GPTMini(V=8, d=32, h=4, N=2, n_maks=64)
    r = latih_gpt(model, lambda: torch.tensor(bangkit_markov(T, 32, 65, rng)),
                  600, 3e-3)
    r = np.convolve(r, np.ones(20) / 20, mode="valid")
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    ax.plot(np.arange(len(r)) + 20, r, color=BIRU, lw=1,
            label="GPTMini (latih)")
    for v, t, ls in ((np.log(8), r"$\log V$", ":"),
                     (float(-(pi * np.log(pi)).sum()), "unigram", "-."),
                     (laju_entropi(T), "laju entropi $H$", "--")):
        ax.axhline(v, color=ABU, lw=0.8, ls=ls)
        ax.text(600, v + 0.03, t, fontsize=5.8, ha="right", color=ABU)
    ax.set_xlabel("langkah")
    ax.set_ylabel("loss (rata-rata bergerak 20)")
    ax.set_ylim(1.2, 2.2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-markov")


def bab11_mlm():
    fig, ax = plt.subplots(figsize=(4.8, 1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    asli = ["ilmu", "itu", "cahaya", "dan", "cahaya", "itu", "terang"]
    masuk = ["ilmu", "[MASK]", "cahaya", "dan", "pelita", "itu", "terang"]
    jenis = [0, 1, 0, 0, 2, 0, 0]
    for i, (a, m, j) in enumerate(zip(asli, masuk, jenis)):
        x = 0.8 + i * 1.35
        wr, isi = ((ABU, "white"), (MERAH, MERAH_MUDA),
                   (JINGGA, JINGGA_MUDA))[j]
        _kotak(ax, x, 2.2, m, w=1.2, h=0.55, wr=wr, isi=isi, fs=5.8)
        if j:
            _panah(ax, (x, 1.88), (x, 1.15), wr=wr)
            ax.text(x, 0.75, a, fontsize=6, ha="center", color=wr)
    ax.text(0.1, 0.2, "merah: diganti [MASK]; jingga: diganti token acak;"
            " loss hanya di posisi yang dipilih", fontsize=5.6, color=ABU)
    simpan(fig, "bab11-mlm")


def bab11_smoothing():
    import torch
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.9))
    V = 4
    for eps, w in ((0.0, MERAH), (0.1, BIRU)):
        z = torch.zeros(V, requires_grad=True)
        opt = torch.optim.SGD([z], lr=1.0)
        celah = []
        for _ in range(400):
            loss = torch.nn.functional.cross_entropy(
                z[None], torch.tensor([0]), label_smoothing=eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            celah.append((z[0] - z[1:].max()).item())
        axs[0].plot(celah, color=w, lw=1,
                    label=f"$\\varepsilon = {angka_mat(eps, 1)}$")
        p = torch.softmax(z.detach(), 0).numpy()
        axs[1].bar(np.arange(V) + (0.2 if eps else -0.2), p, width=0.4,
                   color=w)
    axs[0].axhline(np.log(0.925 / 0.025), color=ABU_GARIS, ls="--", lw=0.8)
    axs[0].set_xlabel("langkah gradient descent")
    axs[0].set_ylabel("selisih logit sasaran")
    axs[0].legend(fontsize=5.5)
    axs[1].set_xticks(range(V))
    axs[1].set_xticklabels(["sasaran", "lain", "lain", "lain"],
                           fontsize=5.5)
    kunci_label(axs[1], "x")
    axs[1].set_ylabel("peluang akhir")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-smoothing")


# ============================ Bab 12 =================================

def bab12_lintasan():
    from bab12_optimisasi import Adam
    f = lambda w: 0.5 * (100 * w[0] ** 2 + w[1] ** 2)
    gr = lambda w: np.array([100 * w[0], w[1]])
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    X, Y = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
    ax.contour(X, Y, 0.5 * (100 * X ** 2 + Y ** 2), levels=[0.5, 2, 8, 30],
               colors=ABU_GARIS, linewidths=0.5)
    for nama, w0, lr, warna in (("GD, $\\eta = 0{,}019$", "gd", 0.019, MERAH),
                                ("Adam, $\\eta = 0{,}05$", "adam", 0.05, BIRU)):
        w = np.array([1.0, 1.0])
        jalan = [w.copy()]
        a = Adam(lr=lr)
        for _ in range(60):
            w = w - lr * gr(w) if w0 == "gd" else a.langkah(w, gr(w))
            jalan.append(w.copy())
        j = np.array(jalan)
        ax.plot(j[:, 0], j[:, 1], color=warna, lw=0.8, marker="o", ms=1.5,
                label=nama)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.1)
    ax.set_xlabel("$w_1$ (kelengkungan 100)")
    ax.set_ylabel("$w_2$ (kelengkungan 1)")
    ax.legend(fontsize=5.5, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-lintasan")


def bab12_jadwal():
    from bab12_optimisasi import laju_belajar
    t = np.arange(1000)
    fig, ax = plt.subplots(figsize=(4.2, 1.7))
    ax.plot(t, [laju_belajar(i, 1e-3, 100, 1000) * 1e3 for i in t],
            color=BIRU, lw=1)
    ax.axvline(100, color=ABU_GARIS, ls="--", lw=0.7)
    ax.text(110, 0.3, "akhir warmup", fontsize=5.8)
    ax.set_xlabel("langkah $t$")
    ax.set_ylabel(r"$\eta_t$ ($\times 10^{-3}$)")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-jadwal")


def bab12_sapuan():
    from bab12_optimisasi import latih_jadwal
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    for kw, w, t in ((dict(opt="sgd"), MERAH, "SGD"),
                     (dict(), JINGGA, "AdamW konstan"),
                     (dict(pemanasan=30), BIRU, "AdamW warmup + kosinus")):
        ax.plot(lrs, [latih_jadwal(lr, **kw)[0] for lr in lrs], color=w,
                marker="o", ms=2.5, lw=0.9, label=t)
    ax.axhline(1.3568, color=ABU_GARIS, ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("laju belajar puncak")
    ax.set_ylabel("loss akhir")
    ax.set_ylim(1.3, 2.1)
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-sapuan")


# ============================ Bab 16 =================================

def bab16_patch():
    from sklearn.datasets import load_digits
    from bab16_luarteks import ke_patch
    img = load_digits().images[0] / 16.0
    P = ke_patch(img[None], 2)[0]
    fig = plt.figure(figsize=(4.8, 1.9))
    ax = fig.add_axes([0.02, 0.1, 0.3, 0.8])
    ax.imshow(img, cmap="Greys")
    for k in (1.5, 3.5, 5.5):
        ax.axhline(k, color=JINGGA, lw=1)
        ax.axvline(k, color=JINGGA, lw=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("gambar 8 x 8, patch 2 x 2", fontsize=6.5)
    ax2 = fig.add_axes([0.38, 0.3, 0.6, 0.45])
    ax2.imshow(P.T, cmap="Greys", aspect="auto")
    ax2.set_xticks(range(16))
    ax2.set_xticklabels([str(i + 1) for i in range(16)], fontsize=5)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(["1", "2", "3", "4"], fontsize=5)
    kunci_label(ax2, "x", "y")
    ax2.set_xlabel("token (patch) ke-$i$, urut baris", fontsize=6)
    ax2.set_ylabel("piksel", fontsize=6)
    ax2.set_title("16 token berdimensi 4", fontsize=6.5)
    simpan(fig, "bab16-patch")


def bab16_posisi():
    from bab16_luarteks import latih_vit
    _, m = latih_vit(model=True)
    E = m.pos.detach().numpy()[0, 1:]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    C = E @ E.T
    fig, axs = plt.subplots(4, 4, figsize=(3.4, 3.4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(C[i].reshape(4, 4), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.plot(i % 4, i // 4, marker="s", ms=3, mfc="none", mec=JINGGA)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    simpan(fig, "bab16-posisi")


def bab16_deret():
    import torch
    from bab16_luarteks import DeretMini, ar2
    rng = np.random.default_rng(BENIH)
    torch.manual_seed(BENIH)
    uji = ar2(64, 500, rng)
    model = DeretMini()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(1500):
        yb = torch.tensor(ar2(64, 64, rng), dtype=torch.float32)
        loss = ((model(yb[:, :-1]) - yb[:, 1:]) ** 2)[:, 2:].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = model(torch.tensor(uji[:1, :-1], dtype=torch.float32)).numpy()[0]
    y = uji[0]
    ar = 4 + 0.5 * (y[1:-1] - 4) - 0.5 * (y[:-2] - 4)
    t = np.arange(64)
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    ax.plot(t, y, color=ABU, lw=0.8, marker="o", ms=2, label="data")
    ax.plot(t[2:], ar, color=BIRU, lw=0.9, label="ramalan AR(2) benar")
    ax.plot(t[3:], p[2:], color=JINGGA, lw=0.9, ls="--",
            label="ramalan Transformer")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y_t$")
    ax.legend(fontsize=5.3, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab16-deret")


# ============================ Bab 13 =================================

def _log13(nama):
    import json
    from bab13_korpus import MODEL
    return json.loads((MODEL / f"{nama}.json").read_text())


def bab13_kurva():
    k, b = _log13("karakter"), _log13("bpe")
    ln2 = np.log(2)
    fb = b["token_val"] / b["karakter_val"] / ln2
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    ax.plot(k["langkah"], np.array(k["latih"]) / ln2, color=BIRU, lw=0.9,
            ls="--", label="karakter, latih")
    ax.plot(k["langkah"], np.array(k["val"]) / ln2, color=BIRU, lw=1.1,
            marker="o", ms=2, label="karakter, validasi")
    lt = np.array(b["latih"]) * fb
    ax.plot(b["langkah"], lt, color=JINGGA, lw=0.9, ls="--",
            label="BPE, latih")
    ax.plot(b["langkah"], np.array(b["val"]) * fb, color=JINGGA, lw=1.1,
            marker="s", ms=2, label="BPE, validasi")
    ax.set_xlabel("langkah")
    ax.set_ylabel("bit per karakter")
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=5.3, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-kurva")


def bab13_overfit():
    u, k = _log13("uud"), _log13("karakter")
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    ax.plot(u["langkah"], u["latih"], color=MERAH, lw=0.9, ls="--",
            label="UUD saja, latih")
    ax.plot(u["langkah"], u["val"], color=MERAH, lw=1.1, marker="o", ms=2,
            label="UUD saja, validasi")
    ax.plot(k["langkah"], k["val"], color=BIRU, lw=1.1, marker="o", ms=2,
            label="sembilan peraturan, validasi")
    j = int(np.argmin(u["val"]))
    ax.annotate("validasi terkecil", (u["langkah"][j], u["val"][j]),
                xytext=(20, 25), textcoords="offset points", fontsize=5.5,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=ABU))
    ax.set_xlabel("langkah")
    ax.set_ylabel("loss (nat per karakter)")
    ax.legend(fontsize=5.3)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-overfit")


def bab13_attention():
    import torch
    from bab13_evaluasi import muat_model
    from bab18_riset import telusur
    model, tok, _ = muat_model("karakter")
    teks = "Pasal 31 (1) Setiap warga negara berhak mendapat"
    ids = torch.tensor(tok.encode(teks))[None]
    _, A, _ = telusur(model, ids)
    fig, axs = plt.subplots(1, 4, figsize=(4.8, 1.55))
    for j, ax in enumerate(axs):
        M = A[0][0, j].numpy()
        ax.imshow(M[-24:, -24:], cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"lapisan 1, head {j + 1}", fontsize=5.8)
    fig.tight_layout(pad=0.3)
    simpan(fig, "bab13-attention")


# ============================ Bab 14 =================================

def bab14_saring():
    from bab01_data import KOSAKATA, data_mini, lintasan_maju
    from bab14_decoding import saring
    z = lintasan_maju(data_mini())["logit"][2]
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    sel = [("suhu 1", {}), ("suhu 0,5", dict(suhu=0.5)),
           ("suhu 2", dict(suhu=2.0)), ("top-p 0,7", dict(p=0.7))]
    w = [ABU, BIRU, JINGGA, HIJAU]
    for j, ((nama, kw), wr) in enumerate(zip(sel, w)):
        ax.bar(np.arange(4) + (j - 1.5) * 0.2, saring(z, **kw), width=0.2,
               color=wr, label=nama)
    ax.set_xticks(range(4))
    ax.set_xticklabels(KOSAKATA)
    kunci_label(ax, "x")
    ax.set_ylabel("peluang token posisi 3")
    ax.legend(fontsize=5.5, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-saring")


def bab14_keragaman():
    from bab14_decoding import keragaman
    h = keragaman()
    fig, ax = plt.subplots(figsize=(4.2, 2.1))
    for nama, lp, nyata, unik in h:
        ax.plot(unik, nyata, marker="o", ms=4, color=BIRU)
        ax.annotate(nama.replace(".", ","), (unik, nyata),
                    xytext=(4, 3), textcoords="offset points", fontsize=5.8)
    ax.set_xlabel("proporsi 4-gram unik (keragaman)")
    ax.set_ylabel("proporsi kata nyata")
    ax.set_xlim(0.5, 0.76)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-keragaman")


def bab14_beam():
    fig, ax = plt.subplots(figsize=(4.8, 2.2))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5)
    ax.axis("off")
    _kotak(ax, 1.0, 2.5, "awal", w=1.2, h=0.6)
    simpul = [((4.0, 3.8), "A", "0,6"), ((4.0, 1.2), "B", "0,4")]
    for (x, y), t, p in simpul:
        _kotak(ax, x, y, t, w=0.8, h=0.55)
        _panah(ax, (1.6, 2.5), (x - 0.42, y))
        ax.text((1.6 + x) / 2, (2.5 + y) / 2 + 0.15, p, fontsize=6,
                color=ABU)
    daun = [((4.0, 3.8), [("AA", "0,4", 0.24), ("AB", "0,3", 0.18),
                          ("AC", "0,3", 0.18)]),
            ((4.0, 1.2), [("BA", "0,9", 0.36), ("BB", "0,05", 0.02),
                          ("BC", "0,05", 0.02)])]
    for (x, y), anak in daun:
        for k, (t, p, tot) in enumerate(anak):
            yy = y + 0.7 - k * 0.7
            warna = MERAH if t == "BA" else (JINGGA if t == "AA" else ABU)
            _kotak(ax, 7.4, yy, f"{t}: {angka(tot, 2)}", w=1.9, h=0.5,
                   wr=warna, isi="white", fs=5.8)
            _panah(ax, (x + 0.42, y), (6.42, yy))
            ax.text(5.3, (y + yy) / 2 + 0.08, p, fontsize=5.5, color=ABU)
    ax.text(8.45, 4.5, "pilihan greedy", fontsize=6, color=JINGGA,
            va="center")
    ax.text(8.45, 1.9, "pilihan beam 2", fontsize=6, color=MERAH,
            va="center")
    simpan(fig, "bab14-beam")


# ============================ Bab 15 =================================

def bab15_lora():
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.add_patch(Rectangle((3.2, 1.0), 2.4, 2.4, facecolor=ABU_GARIS,
                           edgecolor=ABU, lw=0.7))
    ax.text(4.4, 2.2, "$\\mathbf{W}$\n$d \\times d$\n(beku)", ha="center",
            va="center", fontsize=6.5)
    ax.add_patch(Rectangle((6.6, 1.0), 0.45, 2.4, facecolor=JINGGA_MUDA,
                           edgecolor=JINGGA, lw=0.7))
    ax.text(6.82, 0.65, "$\\mathbf{A}$\n$d \\times r$", ha="center",
            va="top", fontsize=6)
    ax.add_patch(Rectangle((7.4, 2.95), 2.4, 0.45, facecolor=JINGGA_MUDA,
                           edgecolor=JINGGA, lw=0.7))
    ax.text(8.6, 3.6, "$\\mathbf{B}$: $r \\times d$, awal 0", ha="center",
            fontsize=6)
    ax.text(6.1, 2.2, "+", fontsize=10, ha="center", va="center")
    ax.text(1.4, 2.2, "$\\mathbf{x}$", fontsize=9, ha="center", va="center")
    _panah(ax, (1.7, 2.2), (3.1, 2.2))
    ax.text(5.0, 4.5, "$\\mathbf{x}^\\top(\\mathbf{W} + \\frac{\\alpha}{r}"
            "\\mathbf{A}\\mathbf{B})$: hanya $\\mathbf{A}$ dan $\\mathbf{B}$"
            " yang dilatih", ha="center", fontsize=6.5, color=ABU)
    simpan(fig, "bab15-lora")


def bab15_data():
    from bab15_finetuning import jalankan
    pk = [20, 50, 100, 400]
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    for c, w, t in (("nol", ABU, "dari nol"), ("probe", HIJAU, "linear probe"),
                    ("penuh", BIRU, "fine-tuning penuh"),
                    ("lora", JINGGA, "LoRA $r = 4$")):
        ax.plot(pk, [jalankan(c, per_kelas=k)[0] for k in pk], color=w,
                marker="o", ms=2.5, lw=0.9, label=t)
    ax.axhline(1 / 9, color=ABU_GARIS, ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(pk)
    ax.set_xticklabels([str(k) for k in pk])
    kunci_label(ax, "x")
    ax.set_xlabel("contoh latih per kelas")
    ax.set_ylabel("akurasi validasi")
    ax.legend(fontsize=5.3, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-data")


def bab15_bingung():
    from bab15_finetuning import jalankan
    from bab13_korpus import muat_korpus
    _, pred, y = jalankan("lora", per_kelas=400, tebakan=True)
    nama = [n.replace("uud1945", "UUD").replace("uu", "UU ")
            for n in muat_korpus()]
    C = np.zeros((9, 9))
    np.add.at(C, (y, pred), 1)
    C = C / C.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if C[i, j] >= 0.05:
                ax.text(j, i, angka(C[i, j], 2)[1:], ha="center",
                        va="center", fontsize=4.8,
                        color="white" if C[i, j] > 0.5 else ABU)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(nama, rotation=60, fontsize=5)
    ax.set_yticklabels(nama, fontsize=5)
    kunci_label(ax, "x", "y")
    ax.set_xlabel("tebakan", fontsize=6)
    ax.set_ylabel("sebenarnya", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab15-bingung")


# ============================ Bab 18 =================================

def bab18_salin():
    import torch
    from bab18_riset import latih_salin, nll_token, salin_batch, telusur
    sm = latih_salin()
    g = torch.Generator().manual_seed(7)
    x = salin_batch(500, 32, 16, g)
    _, A, _ = telusur(sm, x[:, :-1])
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.75),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    for ax, (l, h) in zip(axs[:2], ((0, 3), (1, 0))):
        ax.imshow(A[l][0, h].numpy(), cmap="Greens", vmin=0, vmax=1)
        ax.set_title(f"lapisan {l + 1}, head {h + 1}", fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
    r = []
    for l in range(2):
        for h in range(4):
            nm = nll_token(telusur(sm, x[:, :-1], [(l, h)])[2], x)
            r.append(nm[:, 16:].mean().item())
    w = [MERAH if v > 1 else BIRU for v in r]
    axs[2].bar(range(8), r, color=w)
    axs[2].set_xticks(range(8))
    axs[2].set_xticklabels([f"{l}.{h}" for l in (1, 2) for h in range(1, 5)],
                           fontsize=5)
    kunci_label(axs[2], "x")
    axs[2].set_xlabel("head yang dimatikan", fontsize=6)
    axs[2].set_ylabel("loss salinan", fontsize=6)
    _rapikan(axs[2])
    fig.tight_layout()
    simpan(fig, "bab18-salin")


def bab18_jarak():
    import torch
    from bab18_riset import latih_salin, nll_token, salin_ulang, telusur
    sm = latih_salin()
    g = torch.Generator().manual_seed(7)
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    for k, w in ((16, BIRU), (12, MERAH), (8, JINGGA)):
        x = salin_ulang(500, k, 32, 16, g)
        nk = nll_token(telusur(sm, x[:, :-1])[2], x).mean(0).numpy()
        ax.plot(np.arange(2, 33), nk, color=w, lw=1, marker="o", ms=2,
                label=f"segmen $k = {k}$ diulang")
    ax.axhline(np.log(16), color=ABU_GARIS, ls="--", lw=0.7)
    ax.set_xlabel("posisi token sasaran")
    ax.set_ylabel("loss rata-rata")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab18-jarak")


def bab18_probing():
    import subprocess
    keluaran = subprocess.run([sys.executable, "bab18_riset.py"],
                              cwd="kode", capture_output=True,
                              text=True).stdout
    baris = {b.split(":")[0].strip(): [float(v) for v in b.split(":")[1].split()]
             for b in keluaran.splitlines()
             if b.strip().startswith(("terlatih", "acak"))}
    mayor = float(keluaran.split("terbanyak:")[1].split()[0])
    lap = np.arange(5)
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.plot(lap, baris["terlatih"], color=BIRU, marker="o", ms=3,
            label="model terlatih")
    ax.plot(lap, baris["acak"], color=ABU, marker="s", ms=3,
            label="model acak (kendali)")
    ax.axhline(mayor, color=ABU_GARIS, ls="--", lw=0.7)
    ax.text(4, mayor + 0.02, "kelas terbanyak", fontsize=5.5, ha="right")
    ax.set_xticks(lap)
    ax.set_xticklabels(["embedding", "1", "2", "3", "4"])
    kunci_label(ax, "x")
    ax.set_xlabel("representasi sesudah lapisan")
    ax.set_ylabel("akurasi probe")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5.5, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab18-probing")


# ============================ Bab 17 =================================

def bab17_daring():
    rng = np.random.default_rng(BENIH)
    s = np.sort(rng.normal(size=12) * 1.5)
    s = rng.permutation(s)
    m, l, ms, ls = -np.inf, 0.0, [], []
    for i in range(0, 12, 3):
        b = s[i:i + 3]
        mb = max(m, b.max())
        l = l * np.exp(m - mb) + np.exp(b - mb).sum()
        m = mb
        ms.append(m)
        ls.append(l)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.9))
    axs[0].bar(range(12), s, color=[BIRU, JINGGA, HIJAU, MERAH][0:4] * 3)
    for i in range(4):
        axs[0].bar(range(3 * i, 3 * i + 3), s[3 * i:3 * i + 3],
                   color=[BIRU, JINGGA, HIJAU, MERAH][i])
    axs[0].set_xlabel("key (empat blok)")
    axs[0].set_ylabel("skor")
    axs[1].plot(range(1, 5), ms, color=BIRU, marker="o", ms=3,
                label="maksimum $m$")
    axs[1].plot(range(1, 5), np.log(ls) + ms, color=JINGGA, marker="s",
                ms=3, label=r"$m + \log\ell$")
    axs[1].axhline(np.log(np.exp(s).sum()), color=ABU_GARIS, ls="--",
                   lw=0.8)
    axs[1].set_xticks(range(1, 5))
    kunci_label(axs[1], "x")
    axs[1].set_xlabel("blok yang sudah dibaca")
    axs[1].legend(fontsize=5.5)
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab17-daring")


def bab17_chinchilla():
    from bab17_skala import chinchilla, optimal
    N = np.logspace(8, 12.5, 200)
    D = np.logspace(9, 13.5, 200)
    NN, DD = np.meshgrid(N, D)
    L = chinchilla(NN, DD)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    cs = ax.contour(NN, DD, L, levels=[1.9, 2.0, 2.2, 2.5, 3.0],
                    colors=ABU, linewidths=0.5)
    ax.clabel(cs, fmt=lambda v: f"{v:.1f}".replace(".", ","), fontsize=5)
    C = np.logspace(18, 26, 50)
    opt = np.array([optimal(c) for c in C])
    ax.plot(opt[:, 0], opt[:, 1], color=BIRU, lw=1,
            label="optimum (tetapan pendekatan 3)")
    ax.plot(N, 20 * N, color=HIJAU, lw=0.8, ls="--", label="$D = 20N$")
    for n_, d_, t in ((7e10, 1.4e12, "Chinchilla"), (2.8e11, 3e11, "Gopher")):
        ax.plot(n_, d_, marker="o", ms=3.5, color=MERAH)
        ax.text(n_ * 1.3, d_, t, fontsize=5.8, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameter $N$")
    ax.set_ylabel("token latih $D$")
    ax.set_xlim(1e8, 3e12)
    ax.set_ylim(1e9, 3e13)
    ax.legend(fontsize=5.3, loc="lower right")
    fig.tight_layout()
    simpan(fig, "bab17-chinchilla")


def bab17_sapuan():
    from scipy.optimize import curve_fit
    from bab17_skala import latih_ukuran
    h = [latih_ukuran(d, N) for d, N in ((32, 1), (48, 2), (64, 2),
                                         (96, 3), (128, 4))]
    P, L = map(np.array, zip(*h))
    f = lambda x, E, A, a: E + A * x ** (-a)
    (E, A, a), _ = curve_fit(f, P, L, p0=(0.8, 50, 0.4), maxfev=20000)
    x = np.logspace(4, 6.3, 100)
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.plot(P, L, "o", color=BIRU, ms=4, label="model karakter")
    ax.plot(x, f(x, E, A, a), color=JINGGA, lw=0.9,
            label=f"$E + AP^{{-a}}$, $a = {angka_mat(a, 2)}$")
    ax.axhline(E, color=ABU_GARIS, ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("parameter tanpa embedding $P$")
    ax.set_ylabel("loss validasi")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab17-sapuan")


if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
