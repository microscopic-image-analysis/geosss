"""
Some common distributions on the sphere.
"""

import functools
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import i0, iv, ive, logsumexp

from geosss import sphere
from geosss.spherical_curve import SphericalCurve
from geosss.utils import counter


class Distribution(ABC):
    """Distributions abstract class"""

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass


class Uniform(Distribution):
    def log_prob(self, x):
        return 0.0

    def gradient(self, x):
        return np.zeros_like(x)


@counter(["log_prob", "gradient"])
class Bingham(Distribution):
    r"""Bingham distribution

    (x) \propto \exp(x^T A x)

    where A is a symmetric matrix without loss of generality.

    We can use a spectral decomposition to write
    \[
    A = U\Lambda U^T
    \]
    where $\Lambda$ is diagonal matrix of eigenvalues, and the columns of $U$
    are the eigenvectors of $A$. Moreover, $U$ is a column-orthogonal matrix:
    $U^T U = I$.

    Variable transformation:

    x -> y = U^Tx, x = Uy

    The transformed distribution is

    y \sim \exp(\sum_i \lambda_i y_i^2) = \prod_i \exp(\lambda_i y_i^2)

    subject to $\|y\| = 1$

    \exp(x^T A x) : Bingham distribution
    \exp(x^T b) : von Mises - Fisher distribution
    \exp(x^T A x + x^T b): Fisher-Bingham distribution
    """

    def __init__(self, A):
        assert np.allclose(A, A.T)

        self.A = A

        # eigh returns eigenvalues/-vectors in ascending order
        v, U = np.linalg.eigh(A)

        # descending order
        self.v, self.U = v[::-1], U[:, ::-1]

    def log_prob(self, x):
        """
        Evaluate the log pdf of the Bingham distribution for a single point on
        the sphere (if np.ndim(x) == 1) or multiple points (rows of a rank-2
        array).
        """
        assert x.ndim in (1, 2) and x.shape[-1] == self.d

        return np.sum((x @ self.A) * x, axis=-1)

    def gradient(self, x):
        return 2 * self.A @ x

    @property
    def d(self):
        """Dimension of ambient space."""
        return len(self.A)

    @property
    def mode(self):
        """Point with maximum probability."""
        return self.U[:, 0]

    @property
    def max_log_prob(self):
        return self.v[0]


@counter(["log_prob", "gradient"])
class BinghamFisher(Bingham):
    def __init__(self, A, b):
        super().__init__(A)
        assert len(A) == len(b)
        self.b = b

    def log_prob(self, x):
        return super().log_prob(x) + x @ self.b


@counter(["log_prob", "gradient"])
class VonMisesFisher(Distribution):
    """Von Mises-Fisher distribution

    vMF(x) propto etr(mu^T x)

    where x is restricted to the unit sphere
    """

    def __init__(self, mu):
        self.mu = np.array(mu)

    @property
    def d(self):
        return self.mu.size

    @property
    def kappa(self):
        return np.linalg.norm(self.mu)

    @property
    def mode(self):
        return sphere.radial_projection(self.mu)

    @property
    def max_log_prob(self):
        return self.kappa

    @functools.cached_property
    def log_Z(self):
        d = self.d
        kappa = np.linalg.norm(self.mu)
        log_Z = (
            (d / 2) * np.log(2 * np.pi)
            + np.log(ive(d / 2 - 1, kappa))
            - (d / 2 - 2) * np.log(kappa)
        )
        return log_Z

    def log_prob(self, x):
        return x @ self.mu - np.log(2 * np.pi) - np.log(i0(np.linalg.norm(self.mu)))

    def gradient(self, x):
        return self.mu


@counter("log_prob")
class MarginalVonMisesFisher(VonMisesFisher):
    """Computing marginals of the von Mises-Fisher distribution"""

    def __init__(self, dim_idx, mu):
        super().__init__(mu)
        self.dim_idx = dim_idx

    def prob(self, x):
        d = len(self.mu)
        kappa = np.linalg.norm(self.mu)
        mu = self.mu[self.dim_idx] / kappa
        prob = (
            np.sqrt(kappa / (2 * np.pi))
            / iv(d / 2 - 1, kappa)
            * ((1 - x**2) / (1 - mu**2)) ** ((d - 3) / 4)
            * np.exp(kappa * mu * x)
            * iv((d - 3) / 2, kappa * np.sqrt((1 - mu**2) * (1 - x**2)))
        )
        return prob

    def log_prob(self, x):
        return np.log(np.clip(self.prob(x), 1e-308, None))


@counter("log_prob")
class MultivariateNormal(Distribution):
    def __init__(self, mu, C):
        self.mu = mu
        self.C = C
        self.invC = np.linalg.inv(C)

    def log_prob(self, x):
        return -0.5 * np.sum((x - self.mu) * ((x - self.mu) @ self.invC), axis=-1)


@counter("log_prob")
class ACG(MultivariateNormal):
    def __init__(self, C):
        super().__init__(np.zeros(len(C)), C)

    def log_prob(self, x):
        y = -2 * super().log_prob(x)
        return -np.log(y) * len(self.C) / 2


@counter(["log_prob", "gradient"])
class MixtureModel(Distribution):
    def __init__(self, components, weights=None):
        self.pdfs = components
        if weights is None:
            weights = np.ones(len(components))
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()

    def log_prob(self, x):
        p = np.transpose([pdf.log_prob(x) for pdf in self.pdfs])
        p += np.log(self.weights)
        return logsumexp(p, axis=-1)

    def gradient(self, x):
        p = np.transpose([pdf.log_prob(x) for pdf in self.pdfs])
        p += np.log(self.weights)
        g = np.transpose([pdf.gradient(x) for pdf in self.pdfs])
        return np.sum(np.exp(p) * g, axis=1) / np.exp(logsumexp(p, axis=-1))


def random_bingham(d=2, vmax=None, vmin=None, eigensystem=False, seed=None):
    """
    Create a random Bingham distribution where
    Parameters
    ----------
    d : integer >= 2
        Dimension of ambient space
    vmax : float or None
        Optional maximum eigenvalue of the precision matrix.
    vmin : float or None
        Optional minimum eigenvalue of the precision matrix.
    eigensystem : bool
        Flag specifying if Bingham distribution has diagonal precision matrix
        (i.e. we are working in the eigenvasis of A) or not. (Default value:
        False)
    """
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((d, d))
    A = A.T @ A
    v, U = np.linalg.eigh(A)
    if vmin is not None:
        v += vmin - v.min()
    if vmax is not None:
        v *= vmax / v.max()
    if eigensystem:
        U = np.eye(d)
    A = (U * v) @ U.T
    return Bingham(A)


@counter(["log_prob", "gradient"])
class CurvedVonMisesFisher(Distribution):
    def __init__(self, curve: SphericalCurve, kappa: float = 100.0):
        self.curve = curve
        self.kappa = kappa

    @property
    def d(self):
        """dimension of the sphere"""
        return self.curve.knots.shape[-1]

    def log_prob(self, x):
        if x.ndim == 1:
            return self.kappa * (x @ self.curve.find_nearest(x))
        return np.array(list(map(self.log_prob, x)))

    def gradient(self, x):
        return self.kappa * self.curve.find_nearest(x)
