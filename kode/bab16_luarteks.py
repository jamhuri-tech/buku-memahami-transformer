"""Bab 16: Transformer untuk gambar (ViT kecil pada digit 8 x 8 bawaan
scikit-learn) dan untuk deret waktu (AR(2) dari Memahami Deret Waktu).

Dipakai bab-bab berikutnya:
  ke_patch(X, P)            gambar (m, H, W) -> (m, N, P*P)
  ViTMini(...)              encoder pre-LN dengan token [CLS]
  latih_vit(posisi, acak)   akurasi uji
  DeretMini(...)            decoder kausal untuk nilai real
"""
import numpy as np
import torch
import torch.nn as nn


def ke_patch(X, P):
    m, H, W = X.shape
    X = X.reshape(m, H // P, P, W // P, P).transpose(0, 1, 3, 2, 4)
    return X.reshape(m, (H // P) * (W // P), P * P)


class ViTMini(nn.Module):
    def __init__(self, n_patch, dim_patch, d=32, h=4, N=2, kelas=10,
                 posisi=True):
        super().__init__()
        self.embed = nn.Linear(dim_patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n_patch + 1, d) * 0.02) \
            if posisi else None
        lap = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0,
                                         batch_first=True, norm_first=True,
                                         activation="gelu")
        self.enc = nn.TransformerEncoder(lap, N, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.kepala = nn.Linear(d, kelas)

    def forward(self, x):
        z = torch.cat([self.cls.expand(len(x), -1, -1), self.embed(x)], 1)
        if self.pos is not None:
            z = z + self.pos
        return self.kepala(self.ln(self.enc(z))[:, 0])


def data_digit(P=2, benih=20261009):
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.images / 16.0
    rng = np.random.default_rng(benih)
    i = rng.permutation(len(X))
    X, y = X[i], d.target[i]
    return ke_patch(X, P), y


def latih_vit(posisi=True, acak_patch=False, langkah=1500, benih=20261009,
              model=False):
    X, y = data_digit()
    if acak_patch:                        # susunan patch diacak per gambar
        rng = np.random.default_rng(benih + 1)
        X = np.stack([x[rng.permutation(len(x))] for x in X])
    Xl, yl = torch.tensor(X[:1500], dtype=torch.float32), torch.tensor(y[:1500])
    Xu, yu = torch.tensor(X[1500:], dtype=torch.float32), torch.tensor(y[1500:])
    torch.manual_seed(benih)
    m = ViTMini(X.shape[1], X.shape[2], posisi=posisi)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.05)
    g = torch.Generator().manual_seed(benih)
    for _ in range(langkah):
        i = torch.randint(1500, (64,), generator=g)
        loss = nn.functional.cross_entropy(m(Xl[i]), yl[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(Xu).argmax(1) == yu).float().mean().item()
    return (acc, m) if model else acc


class DeretMini(nn.Module):
    """Setiap nilai y_t menjadi token lewat proyeksi linear 1 -> d."""

    def __init__(self, d=32, h=4, N=2, n=64):
        super().__init__()
        self.masuk = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        lap = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lap, N, enable_nested_tensor=False)
        self.keluar = nn.Linear(d, 1)

    def forward(self, y):
        n = y.shape[1]
        m = nn.Transformer.generate_square_subsequent_mask(n)
        z = self.masuk(y[..., None]) + self.pos[:, :n]
        return self.keluar(self.enc(z, mask=m, is_causal=True))[..., 0]


def ar2(T, B, rng, phi=(0.5, -0.5), mu=4.0):
    y = np.zeros((B, T + 50))
    e = rng.normal(size=(B, T + 50))
    for t in range(2, T + 50):
        y[:, t] = phi[0] * y[:, t - 1] + phi[1] * y[:, t - 2] + e[:, t]
    return y[:, 50:] + mu


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    X, y = data_digit()
    print("(1) Digit 8 x 8 scikit-learn: 1797 gambar, patch 2 x 2")
    print(f"    {X.shape[1]} patch per gambar, dimensi patch {X.shape[2]};"
          " latih 1500, uji 297")
    lr = LogisticRegression(max_iter=2000).fit(X[:1500].reshape(1500, -1),
                                               y[:1500])
    acc_lr = lr.score(X[1500:].reshape(297, -1), y[1500:])
    print(f"    regresi logistik pada 64 piksel : akurasi {acc_lr:.3f}")
    for nama, kw in (("ViT dengan posisi", dict()),
                     ("ViT tanpa posisi", dict(posisi=False)),
                     ("ViT, patch diacak", dict(acak_patch=True))):
        print(f"    {nama:17s}               : akurasi"
              f" {latih_vit(**kw):.3f}")

    rng = np.random.default_rng(20261009)
    torch.manual_seed(20261009)
    uji = ar2(64, 500, rng)
    model = DeretMini()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(1500):
        yb = torch.tensor(ar2(64, 64, rng), dtype=torch.float32)
        p = model(yb[:, :-1])
        loss = ((p - yb[:, 1:]) ** 2)[:, 2:].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = model(torch.tensor(uji[:, :-1], dtype=torch.float32)).numpy()
    mse_t = ((p - uji[:, 1:]) ** 2)[:, 2:].mean()
    ramal = 4 + 0.5 * (uji[:, 1:-1] - 4) - 0.5 * (uji[:, :-2] - 4)
    mse_ar = ((ramal - uji[:, 2:]) ** 2).mean()
    naif = ((uji[:, 1:-1] - uji[:, 2:]) ** 2).mean()
    print("(2) AR(2) phi = (0.5, -0.5), sigma^2 = 1, 500 deret uji x 64:")
    print("    MSE ramalan satu langkah:")
    print(f"    naif {naif:.4f}, AR(2) benar {mse_ar:.4f},"
          f" Transformer {mse_t:.4f}")
