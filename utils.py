import numpy as np
from scipy import linalg


def generate_matrix(n, d, t, xi):

    G = np.random.randn(d, d)
    U, _ = linalg.qr(G)
    U = U[:t]
    dd = [1-(i-1)/t for i in range(t)]
    D = np.zeros((t, t))
    np.fill_diagonal(D, dd)
    S = np.random.randn(n, t)
    return S@D@U, S@D@U + np.random.randn(n, d)/ xi