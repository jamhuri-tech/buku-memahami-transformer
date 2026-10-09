"""Bab 17: softmax daring dan attention per blok (gagasan FlashAttention),
attention linear, mixture of experts, dan scaling law.

Dipakai bab-bab berikutnya:
  attn_blok(Q, K, V, b)          attention kausal blok demi blok, tepat
  attn_linear(Q, K, V)           attention linear kausal, phi = elu + 1
  chinchilla(N, D)               loss menurut tetapan Hoffmann dkk. (2022)
  optimal(C)                     (N, D) optimum untuk compute C = 6ND
  latih_ukuran(d, N, langkah)    model karakter kecil untuk sapuan ukuran
"""
import numpy as np

from bab01_data import mask_kausal, softmax

# tetapan pencocokan Hoffmann dkk. (2022), pendekatan 3
E_, A_, B_, ALFA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def attn_blok(Q, K, V, b):
    """Setiap baris query membaca key per blok berukuran b, menyimpan
    maksimum m, penyebut l, dan pembilang o yang terus diperbarui."""
    n, dk = Q.shape
    O = np.zeros((n, V.shape[1]))
    for t in range(n):
        m, l, o = -np.inf, 0.0, np.zeros(V.shape[1])
        for a in range(0, t + 1, b):
            s = K[a:min(a + b, t + 1)] @ Q[t] / np.sqrt(dk)
            m_baru = max(m, s.max())
            koreksi = np.exp(m - m_baru)
            p = np.exp(s - m_baru)
            l = l * koreksi + p.sum()
            o = o * koreksi + p @ V[a:min(a + b, t + 1)]
            m = m_baru
        O[t] = o / l
    return O


def phi(x):
    return np.where(x > 0, x + 1, np.exp(x))


def attn_linear(Q, K, V):
    """o_t = phi(q_t)^T S_t / phi(q_t)^T z_t, S_t = sum_{s<=t} phi(k_s) v_s^T."""
    S = np.zeros((Q.shape[1], V.shape[1]))
    z = np.zeros(Q.shape[1])
    O = np.zeros((len(Q), V.shape[1]))
    for t in range(len(Q)):
        fk = phi(K[t])
        S += np.outer(fk, V[t])
        z += fk
        fq = phi(Q[t])
        O[t] = fq @ S / (fq @ z)
    return O


def chinchilla(N, D):
    return E_ + A_ / N**ALFA + B_ / D**BETA


def optimal(C):
    """N dan D yang meminimumkan chinchilla(N, D) dengan 6 N D = C."""
    G = (ALFA * A_ / (BETA * B_)) ** (1 / (ALFA + BETA))
    N = G * (C / 6) ** (BETA / (ALFA + BETA))
    return N, C / (6 * N)


def latih_ukuran(d, N, langkah=1200, benih=20261009):
    import torch
    from bab10_arsitektur import GPTMini
    from bab12_optimisasi import laju_belajar
    from bab13_korpus import TokenKarakter, bagi, muat_korpus
    from bab13_latih import batch, loss_val
    latih, val = bagi(muat_korpus())
    tok = TokenKarakter(latih + val)
    il, iv = tok.encode(latih), tok.encode(val)
    torch.manual_seed(benih)
    rng = np.random.default_rng(benih)
    m = GPTMini(V=tok.V, d=d, h=4, N=N, n_maks=128)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for t in range(langkah):
        for g in opt.param_groups:
            g["lr"] = laju_belajar(t, 3e-3, 60, langkah, 3e-4)
        x = batch(il, 16, 128, rng)
        z = m(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(z.reshape(-1, tok.V),
                                                 x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    P = sum(p.numel() for n_, p in m.named_parameters()
            if "tok" not in n_ and "pos" not in n_)
    return P, loss_val(m, iv, 128)


if __name__ == "__main__":
    from bab01_data import WK, WQ, WV, data_mini

    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(20261009)
    n, dk = 50, 8
    Q, K, V = (rng.normal(size=(n, dk)) for _ in range(3))
    biasa = softmax(Q @ K.T / np.sqrt(dk) + mask_kausal(n)) @ V
    print("(1) Attention per blok lawan attention biasa, n = 50:")
    for b in (1, 7, 16, 50):
        print(f"    blok {b:2d}: selisih maks"
              f" {np.abs(attn_blok(Q, K, V, b) - biasa).max():.1e}")
    X = data_mini()
    Ol = attn_linear(X @ WQ, X @ WK, X @ WV)
    print("(2) Attention linear data mini, phi(x) = elu(x) + 1:")
    print("    O =", str(Ol).replace("\n", "\n        "))
    print("(3) L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28:")
    for N, D in ((7e10, 1.4e12), (2.8e11, 3e11), (1.75e11, 3e11)):
        print(f"    N {N:.2e}  D {D:.2e}  C = 6ND {6 * N * D:.2e}"
              f"  L {chinchilla(N, D):.4f}")
    print("(4) Alokasi optimum untuk compute C (6ND = C):")
    for C in (1e19, 1e21, 5.88e23, 1e25):
        N, D = optimal(C)
        print(f"    C {C:.2e}  N {N:.2e}  D {D:.2e}  D/N {D / N:5.1f}"
              f"  L {chinchilla(N, D):.4f}")
    from scipy.optimize import curve_fit
    print("(5) Sapuan ukuran model karakter, 1200 langkah, batch 16:")
    print("      d  N    P (tanpa emb.)   loss val")
    hasil = []
    for d, N in ((32, 1), (48, 2), (64, 2), (96, 3), (128, 4)):
        P, L = latih_ukuran(d, N)
        hasil.append((P, L))
        print(f"    {d:3d} {N:2d} {P:14,d} {L:10.4f}")
    P, L = map(np.array, zip(*hasil))
    f = lambda x, E, A, a: E + A * x ** (-a)
    (E, A, a), _ = curve_fit(f, P, L, p0=(0.8, 50, 0.4), maxfev=20000)
    print(f"    pencocokan L = E + A P^-a: E = {E:.3f}, A = {A:.1f},"
          f" a = {a:.3f}")
