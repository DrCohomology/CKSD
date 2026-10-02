"""CKSD test on the flat torus Y = S^1 x S^1 (Sec. 4.1) with a conditional cosine model, vectorised.

Points of the torus are angles theta = (theta_1, theta_2) in [-pi, pi)^2, stored as the rows of an (n, 2) array.
The chart y(theta) = (cos t1, sin t1, cos t2, sin t2) in R^4 has tangent vectors e_1 = (-sin t1, cos t1, 0, 0),
e_2 = (0, 0, -sin t2, cos t2), so G = I_2 and J = sqrt(det G) = 1: the coordinate Stein feature map needs no
volume correction.

The experiment (level, power and runtime against the MMD test) is in torus_plots.py.
"""
import numpy as np
from scipy.special import i0e, ive

from utils import sample_rows
from utils import GaussianKernel1D, wild_bootstrap_pvalue  # noqa: F401  re-exported, so `from torus import ...` still works


def wrap(a):
    """Angles to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def embed(T):
    """y(theta) = (cos t1, sin t1, cos t2, sin t2) for all rows of T (N, 2) -> (N, 4)."""
    c, s = np.cos(T), np.sin(T)
    return np.stack([c[:, 0], s[:, 0], c[:, 1], s[:, 1]], axis=1)


def tangent(T):
    """e_a(theta) = dy/dtheta_a for all rows of T (N, 2) -> (N, 2, 4)."""
    c, s, z = np.cos(T), np.sin(T), np.zeros(len(T))
    e1 = np.stack([-s[:, 0], c[:, 0], z, z], axis=1)
    e2 = np.stack([z, z, -s[:, 1], c[:, 1]], axis=1)
    return np.stack([e1, e2], axis=1)


class TorusGaussianKernel:
    """l(y, y') = exp(gamma y^T y') = exp(gamma (cos D_1 + cos D_2)),  D_a = theta_a - theta'_a.
    Since ||y||^2 = 2, l = e^{2 gamma} exp(-gamma/2 ||y - y'||^2): the Gaussian kernel of R^4 restricted to the torus.

    Parameters
    ----------
    gamma : float, default 1.0
        Concentration of the kernel.
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, A, B):
        """Gram matrix l(A[i], B[j]) of the angles in the rows of A (N, 2) and B (M, 2) -> (N, M)."""
        return np.exp(self.gamma * embed(A) @ embed(B).T)

    def derivatives(self, A, B):
        """l and its coordinate derivatives on all pairs of rows of A (N, 2), B (M, 2):
            d_{theta_a} l                    = gamma e_a(theta)^T y' l                                 -> (N, M, 2)
            d_{theta'_a} l                   = gamma y^T e_a(theta') l                                 -> (N, M, 2)
            sum_a d_{theta_a} d_{theta'_a} l = gamma sum_a [e_a^T e'_a + gamma (e_a^T y')(y^T e'_a)] l  -> (N, M)

        Returns
        -------
        L, dL, dLp, tr : arrays
            l (N, M), d_theta l (N, M, 2), d_theta' l (N, M, 2), sum_a d_{theta_a} d_{theta'_a} l (N, M).
        """
        yA, yB, eA, eB = embed(A), embed(B), tangent(A), tangent(B)
        L = np.exp(self.gamma * yA @ yB.T)
        ey = np.einsum("nak,mk->nma", eA, yB)          # e_a(theta)^T y'        = -sin D_a
        ye = np.einsum("nk,mak->nma", yA, eB)          # y^T e_a(theta')        =  sin D_a
        ee = np.einsum("nak,mak->nma", eA, eB)         # e_a(theta)^T e_a(theta') = cos D_a
        dL = self.gamma * ey * L[:, :, None]
        dLp = self.gamma * ye * L[:, :, None]
        tr = self.gamma * (ee + self.gamma * ey * ye).sum(axis=2) * L
        return L, dL, dLp, tr


class TorusSteinKernel:
    """K_Q((x, y), (x', y')) = k(x, x') <psi_{Q_Y|x}(y), psi_{Q_Y|x'}(y')>, Y = S^1 x S^1, coordinate feature map
    psi_{Q_Y|x}(y)_a = d_{theta_a} log(q_x J) l(y, .) + d_{theta_a} l(y, .),  a = 1, 2,  with J = 1.

    Parameters
    ----------
    k : callable
        Kernel on the covariates, k(x, x') -> Gram matrix (e.g. utils.GaussianKernel1D).
    l : TorusGaussianKernel
        Kernel on the torus; must provide `derivatives`.
    """

    def __init__(self, k, l):
        self.k = k
        self.l = l

    def score(self, grad_log_q, xs, ts):
        """s_q(x, y) = d_theta log(q_x J) = d_theta log q_x, for all samples at once (J = 1 on this chart)."""
        return grad_log_q(xs, ts)

    def gram(self, grad_log_q, xs, ts, xps=None, tps=None):
        """Matrix of K_Q between the samples (xs[i], ts[i]) and (xps[j], tps[j]); the latter default to the former.

        Parameters
        ----------
        grad_log_q : callable
            grad_log_q(xs, ts) = d_theta log q_x(theta) row-wise, (n,), (n, 2) -> (n, 2), e.g. built on `cosine_score`.
            Only the score enters: the normalising constant of the model is never needed.
        xs : (n,) array
            Covariates.
        ts : (n, 2) array
            Angles.
        xps, tps : (m,), (m, 2) arrays, optional
            Second sample; defaults to (xs, ts).

        Returns
        -------
        (n, m) array
            With the defaults, the input of utils.wild_bootstrap_pvalue; its mean is the CKSD V-statistic.
        """
        xps, tps = (xs, ts) if xps is None else (xps, tps)
        s, sp = self.score(grad_log_q, xs, ts), self.score(grad_log_q, xps, tps)
        L, dL, dLp, tr = self.l.derivatives(ts, tps)
        H = (s @ sp.T * L                                  # s^T s' l
             + np.einsum("ia,ija->ij", s, dLp)             # s^T d_{theta'} l
             + np.einsum("ija,ja->ij", dL, sp)             # d_theta l^T s'
             + tr)                                         # sum_a d_{theta_a} d_{theta'_a} l
        return self.k(xs, xps) * H


def cosine_log_q(T, k1, k2, k3, mu, nu):
    """log q(theta) of the cosine model (Mardia, Taylor & Subramaniam, 2007) up to a constant, row-wise parameters:
    k1 cos(t1 - mu) + k2 cos(t2 - nu) - k3 cos(t1 - mu - t2 + nu)."""
    u, v = T[:, 0] - mu, T[:, 1] - nu
    return k1 * np.cos(u) + k2 * np.cos(v) - k3 * np.cos(u - v)


def cosine_score(T, k1, k2, k3, mu, nu):
    """d_theta log q = (-k1 sin u + k3 sin(u - v), -k2 sin v - k3 sin(u - v)),  u = t1 - mu, v = t2 - nu -> (N, 2).

    Parameters
    ----------
    T : (N, 2) array
        Angles.
    k1, k2, k3, mu, nu : floats or (N,) arrays
        Parameters of the cosine model, one set per row of T or shared.
    """
    u, v = T[:, 0] - mu, T[:, 1] - nu
    return np.stack([-k1 * np.sin(u) + k3 * np.sin(u - v),
                     -k2 * np.sin(v) - k3 * np.sin(u - v)], axis=1)


def cosine_log_normalizer(k1, k2, k3, pmax=200):
    """log C^{-1},  C^{-1} = 4 pi^2 sum_{p in Z} I_p(k1) I_p(k2) I_p(-k3)
                           = 4 pi^2 [I_0 I_0 I_0 + 2 sum_{p>=1} (-1)^p I_p(k1) I_p(k2) I_p(k3)].
    Only needed for checks: the score and the Stein kernel do not depend on it."""
    k1, k2, k3 = (np.asarray(k, dtype=float)[..., None] for k in (k1, k2, k3))
    p = np.arange(pmax)
    terms = ive(p, k1) * ive(p, k2) * ive(p, -k3)                  # I_p(k) = ive(p, k) e^{|k|}
    s = terms[..., 0] + 2 * terms[..., 1:].sum(axis=-1)
    return np.log(4 * np.pi ** 2) + (np.abs(k1) + np.abs(k2) + np.abs(k3))[..., 0] + np.log(s)


def sample_cosine(rng, k1, k2, k3, mu, nu, ngrid=1024, chunk=2048):
    """One draw theta ~ cosine model per entry of the (broadcast) parameters -> (N, 2).
    theta_1 from its marginal q(theta_1) ∝ exp(k1 cos u) I_0(R(u)), u = theta_1 - mu, by inverse-CDF on a grid of
    ngrid cells (uniform within the cell); theta_2 | theta_1 ~ vM(nu + atan2(B, A), R) exactly, with
    A = k2 - k3 cos u, B = -k3 sin u, R = sqrt(A^2 + B^2). Processed in chunks to bound memory.

    Parameters
    ----------
    rng : numpy.random.Generator
    k1, k2, k3, mu, nu : floats or (N,) arrays
        Parameters of the cosine model; broadcast against each other, one draw per entry.
    ngrid : int, default 1024
        Number of grid cells for the marginal of theta_1.
    chunk : int, default 2048
        Largest number of draws handled at once.

    Returns
    -------
    (N, 2) array
        Angles in [-pi, pi).
    """
    k1, k2, k3, mu, nu = np.broadcast_arrays(*(np.atleast_1d(np.asarray(p, dtype=float)) for p in (k1, k2, k3, mu, nu)))
    if len(k1) > chunk:
        return np.concatenate([sample_cosine(rng, *(p[i:i + chunk] for p in (k1, k2, k3, mu, nu)), ngrid=ngrid,
                                             chunk=chunk) for i in range(0, len(k1), chunk)])
    h = 2 * np.pi / ngrid
    u = -np.pi + h * (np.arange(ngrid) + 0.5)                                  # cell midpoints
    R = np.sqrt(k2[:, None] ** 2 + k3[:, None] ** 2 - 2 * k2[:, None] * k3[:, None] * np.cos(u))
    logm = k1[:, None] * np.cos(u) + np.log(i0e(R)) + R                       # log q(theta_1 - mu) + const
    P = np.exp(logm - logm.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    u = u[sample_rows(rng, P)] + h * (rng.random(len(k1)) - 0.5)
    A, B = k2 - k3 * np.cos(u), -k3 * np.sin(u)
    v = np.arctan2(B, A) + rng.vonmises(0.0, np.hypot(A, B))
    return wrap(np.stack([mu + u, nu + v], axis=1))
