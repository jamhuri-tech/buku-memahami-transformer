"""Bab 1: data mini dan satu lintasan maju Transformer satu lapisan.

Dipakai bab-bab berikutnya:
  KOSAKATA, E          kosakata empat token dan embedding 4 x 2
  MASUK, SASARAN       id token masukan dan sasaran kalimat mini
  data_mini()          X (3 x 2), satu baris per token masukan
  WQ, WK, WV, W1, W2   bobot satu head dan FFN data mini
  softmax(Z)           softmax per baris, stabil secara numerik
  mask_kausal(n)       matriks M: 0 di bawah diagonal, -inf di atasnya
  lintasan_maju(X)     semua besaran antara, dalam satu dict
"""
import numpy as np

BENIH = 20261009

KOSAKATA = ["<awal>", "ilmu", "itu", "cahaya"]
# Baris ke-i: embedding token ke-i. Baris = token, kolom = dimensi.
E = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0],
              [-1.0, 1.0]])

# "ilmu itu cahaya": masukan <awal> ilmu itu, sasaran ilmu itu cahaya
MASUK = [0, 1, 2]
SASARAN = [1, 2, 3]

WQ = np.array([[1.0, 0.0], [1.0, 1.0]])
WK = np.array([[0.0, 1.0], [1.0, 0.0]])
WV = np.array([[1.0, 1.0], [-1.0, 1.0]])
# FFN d -> 4 -> d, tanpa bias
W1 = np.array([[1.0, 0.0, 1.0, -1.0],
               [0.0, 1.0, 1.0, 1.0]])
W2 = 0.5 * np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [-1.0, 0.0],
                     [0.0, -1.0]])


def data_mini():
    """X = baris-baris E untuk token masukan."""
    return E[MASUK].copy()


def one_hot(ids, V=len(KOSAKATA)):
    O = np.zeros((len(ids), V))
    O[np.arange(len(ids)), ids] = 1.0
    return O


def softmax(Z, axis=-1):
    """exp(z - maks) / jumlah; pengurangan maks tidak mengubah hasil."""
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.max(axis=axis, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=axis, keepdims=True)


def mask_kausal(n):
    """M[t, s] = 0 bila s <= t, -inf bila s > t."""
    M = np.zeros((n, n))
    M[np.triu_indices(n, 1)] = -np.inf
    return M


def lintasan_maju(X, WQ=WQ, WK=WK, WV=WV, W1=W1, W2=W2, E=E,
                  kausal=True):
    """Satu lapisan decoder tanpa normalisasi; W_O = I."""
    n = X.shape[0]
    Q, K, V = X @ WQ, X @ WK, X @ WV
    dk = K.shape[1]
    S = Q @ K.T / np.sqrt(dk)
    M = mask_kausal(n) if kausal else np.zeros((n, n))
    A = softmax(S + M)
    O = A @ V
    H = X + O
    Z = np.maximum(H @ W1, 0.0)
    F = Z @ W2
    H2 = H + F
    logit = H2 @ E.T
    P = softmax(logit)
    return dict(Q=Q, K=K, V=V, S=S, A=A, O=O, H=H, Z=Z, F=F, H2=H2,
                logit=logit, P=P)


def loss_ce(P, sasaran):
    """Cross-entropy rata-rata: -(1/n) sum_t log P[t, y_t]."""
    return -np.mean(np.log(P[np.arange(len(sasaran)), sasaran]))
