import numpy as np
from scipy import linalg


def svs(X, g):
    Y = []
    n, d = X.shape
    U, dd, V = linalg.svd(X)
    p = g(dd*dd)
    print(p)
    w = dd / np.sqrt(p)
    for i in range(min(n, d)):
        if np.random.rand() < p[i]:
            Y.append(w[i]*V[i])
    return np.array(Y)


def fd(X, eps):
    n, m = X.shape
    l = min(min(n, np.int32(np.ceil(1 / eps))), m)
    Y = np.zeros((l, m))
    for i in range(n):
        Y[l-1] = X[i]
        U, d, V = linalg.svd(Y)
        delta = d[l-1]*d[l-1]
        sigma_hat = d*d - delta*np.ones_like(d)
        sigma_hat = np.sqrt((sigma_hat >= 0) * sigma_hat)
        Sigma_hat = np.zeros_like(Y)
        np.fill_diagonal(Sigma_hat, sigma_hat)
        Y = Sigma_hat @ V

    return Y

def rs(X, r):
    n, d = X.shape
    w = np.linalg.norm(X, axis=1)**2
    p = w/np.sum(w)
    index = np.random.choice(np.arange(n), size=r, p=p)
    return X[index] / np.sqrt(r*p[index]).reshape(-1, 1)

def ksvd(X, k):
    U, d, V = linalg.svd(X)
    Sigma = np.zeros((k, k))
    np.fill_diagonal(Sigma, d[:k])
    return Sigma @ V[:k]

class Solver:

    def __init__(self, s, eps, cost=None):
        self.s = s
        self.cost = cost
        self.eps = eps

    def run(self, X):
        raise NotImplemented


class RS(Solver):

    def __init__(self, s, eps, cost):
        super().__init__(s, eps, cost=cost)

    def run(self, X):
        n, d = X.shape
        Y = np.zeros((0, d))
        for i in range(self.s):
            x_sub = X[i*1000:(i+1)*1000].copy()
            y_sub = rs(x_sub, min(self.cost, x_sub.shape[0]))
            Y = np.vstack((Y, y_sub))
        return fd(Y, self.eps) if self.eps else Y


class eFD(Solver):

    def __init__(self, s, eps, cost):
        super().__init__(s, eps, cost=cost)

    def run(self, X):
        n, d = X.shape
        Y = np.zeros((0, d))
        for i in range(self.s):
            x_sub = X[i*1000:(i+1)*1000].copy()
            y_sub = ksvd(x_sub, min(self.cost, x_sub.shape[0]))
            Y = np.vstack((Y, y_sub))
        return fd(Y, self.eps) if self.eps else Y


class SVS(Solver):
    def __init__(self, s, g, eps, cost):
        super().__init__(s, eps, cost=cost)
        self.g = g

    def run(self, X):
        n, d = X.shape
        Y = np.zeros((0, d))
        for i in range(self.s):
            x_sub = X[i*1000:(i+1)*1000].copy()
            y_sub = svs(x_sub, self.g)
            print(self.g.name, y_sub.shape)
            if y_sub.shape[0] == 0:
                continue
            Y = np.vstack((Y, y_sub[:min(y_sub.shape[0], self.cost)]))
        return fd(Y, self.eps) if self.eps else Y





