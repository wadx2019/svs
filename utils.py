import numpy as np
from scipy import linalg
from setting import *


def generate_matrix(t, xi):

    G = np.random.randn(d, d)
    U, _ = linalg.qr(G)
    U = U[:t]
    dd = [1-(i-1)/t for i in range(t)]
    D = np.zeros((t, t))
    np.fill_diagonal(D, dd)
    S = np.random.randn(n, t)
    return S@D@U, S@D@U + np.random.randn(n, d)/xi