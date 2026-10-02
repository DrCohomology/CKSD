"""CKSD on rankings (Sec. 4.2, d = 1): orderings of S_n, Kendall distance, Mallows kernel and the Stein kernel, vectorised.

A ranking is handled through its index j in a list `space` of all the elements of S_n, so the cyclic shift is
(j +- 1) mod |Y| and every evaluation of the kernel l is a lookup in the |Y| x |Y| table L = l(space, space).

The experiment (level, power and runtime against the MMD test) is in permutations_plot.py.
"""
import numpy as np


def sjt(n):
    """Steinhaus-Johnson-Trotter order: consecutive permutations differ by one adjacent swap, cyclically.

    Parameters
    ----------
    n : int
        Number of items.

    Returns
    -------
    list of tuple
        The n! permutations of (0, ..., n - 1), in SJT order.
    """
    if n == 1:
        return [(0,)]
    out = []
    for i, p in enumerate(sjt(n - 1)):
        for k in (range(n - 1, -1, -1) if i % 2 == 0 else range(n)):
            out.append(p[:k] + (n - 1,) + p[k:])
    return out


def gray_space(na):
    """All rankings of na items, ordered so that every step (incl. last -> first) has Kendall distance 1.

    Parameters
    ----------
    na : int
        Number of items.

    Returns
    -------
    (na!, na) int array
        One ranking per row: entry i is the rank of item i.
    """
    return np.array([np.argsort(p) for p in sjt(na)])


def kendall(A, B):
    """Kendall distances between all rows of A (N, na) and B (M, na) -> (N, M).

    The Kendall distance is the number of discordant pairs, i.e. of pairs of items that the two rankings put in
    opposite order; it ranges from 0 to na (na - 1) / 2.
    """
    sA = np.sign(A[:, :, None] - A[:, None, :]).reshape(len(A), -1)
    sB = np.sign(B[:, :, None] - B[:, None, :]).reshape(len(B), -1)
    na = A.shape[1]
    return (na * (na - 1) / 2 - sA @ sB.T / 2) / 2


class MallowsKernel:
    """Mallows kernel on rankings, l(y, y') = exp(-nu d_K(y, y')), with d_K the Kendall distance.

    Parameters
    ----------
    na : int
        Number of items.
    nu : float, optional
        Bandwidth; defaults to 2 / (na (na - 1)), the inverse of the largest Kendall distance.
    """

    def __init__(self, na, nu=None):
        self.nu = nu if nu is not None else 2 / (na * (na - 1))

    def __call__(self, A, B):
        """Gram matrix l(A[i], B[j]) of the rankings in the rows of A (N, na) and B (M, na) -> (N, M)."""
        return np.exp(-self.nu * kendall(A, B))


class YangSteinKernel1D:
    """K_Q((x, y), (x', y')) = k(x, x') <psi_{Q_Y|x}(y), psi_{Q_Y|x'}(y')>, with Y = space, d = 1.

    Discrete Stein feature map (Yang et al., 2018) built on the cyclic shift of the list `space`,
    y = space[j] -> space[j + 1 mod |Y|]:
        psi_{Q_Y|x}(y) = s_q(x, y) l(y, .) - Delta*_y l(y, .),
        s_q(x, y)      = 1 - q_x(shift y) / q_x(y),
        Delta*_y f(y)  = f(y) - f(shift^-1 y).
    The ordering of `space` fixes the shift: any ordering gives a valid test, but the power can depend on it.

    Parameters
    ----------
    k : callable
        Kernel on the covariates, k(x, x') -> Gram matrix (e.g. utils.GaussianKernel1D).
    l : callable
        Kernel on the rankings, l(A, B) -> Gram matrix (e.g. MallowsKernel).
    space : (|Y|, na) int array
        All the rankings, one per row, in the order that defines the shift.
    """

    def __init__(self, k, l, space):
        self.k = k
        self.space = space
        M = len(space)
        self.L = l(space, space)                       # l on all pairs of rankings, computed once
        self.nxt = (np.arange(M) + 1) % M              # shift:      y[j] -> y[j+1]
        self.prv = (np.arange(M) - 1) % M              # inverse:    y[j] -> y[j-1]

    def score(self, log_q, xs, js):
        """s_q(x, y) = 1 - q_x(shift y) / q_x(y), for all samples at once.

        Parameters
        ----------
        log_q : callable
            log_q(xs, js) = log q_x(space[j]) element-wise, up to an additive constant that may depend on x.
        xs : (n,) array
            Covariates.
        js : (n,) int array
            Rankings, as indices into `self.space`.

        Returns
        -------
        (n,) array
        """
        return -np.expm1(log_q(xs, self.nxt[js]) - log_q(xs, js))

    def gram(self, log_q, xs, js):
        """n x n matrix of K_Q over the samples (xs[i], space[js[i]]).

        Parameters
        ----------
        log_q : callable
            log_q(xs, js) = log q_x(space[j]) element-wise, up to an additive constant that may depend on x:
            the normalising constant of the model is never needed.
        xs : (n,) array
            Covariates.
        js : (n,) int array
            Rankings, as indices into `self.space` (not into another ordering of S_n).

        Returns
        -------
        (n, n) array
            Input of utils.wild_bootstrap_pvalue; its mean is the CKSD V-statistic.
        """
        L, s = self.L, self.score(log_q, xs, js)
        a, pa = js, self.prv[js]
        L_ab = L[np.ix_(a, a)]                         # l(y, y')
        L_pb = L[np.ix_(pa, a)]                        # l(shift^-1 y, y')
        L_ap = L[np.ix_(a, pa)]                        # l(y, shift^-1 y')
        L_pp = L[np.ix_(pa, pa)]                       # l(shift^-1 y, shift^-1 y')
        H = (s[:, None] * L_ab * s[None, :]
             - s[:, None] * (L_ab - L_ap)              # s    Delta*_{y'} l
             - (L_ab - L_pb) * s[None, :]              # Delta*_y l    s'
             + L_ab - L_pb - L_ap + L_pp)              # Delta*_{y,y'} l
        return self.k(xs, xs) * H
