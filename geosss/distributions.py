"""
Some common distributions on the sphere.
"""
import functools

import numpy as np
from scipy.special import i0, ive, logsumexp

from . import sphere
from .utils import counter


class Distribution:

    def log_prob(self, x):
        pass


class Uniform(Distribution):

    def log_prob(self, x):
        return 0.

    def gradient(self, x):
        return np.zeros_like(x)


@counter(["log_prob", "gradient"])
class Bingham(Distribution):
    """Bingham distribution

    p(x) \propto \exp(x^T A x) 

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
        """Dimension of ambient space. 
        """
        return len(self.A)

    @property
    def mode(self):
        """Point with maximum probability. 
        """
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

    vMF(x) \propto etr(mu^T x)

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
        log_Z = (d/2) * np.log(2 * np.pi) + np.log(ive(d/2 - 1, kappa)) - \
            (d/2 - 2) * np.log(kappa)
        return log_Z

    def log_prob(self, x):
        return x @ self.mu - np.log(2 * np.pi) \
            - np.log(i0(np.linalg.norm(self.mu)))

    def gradient(self, x):
        return self.mu


@counter("log_prob")
class MultivariateNormal(Distribution):

    def __init__(self, mu, C):
        self.mu = mu
        self.C = C
        self.invC = np.linalg.inv(C)

    def log_prob(self, x):
        return -0.5 * np.sum((x-self.mu) * ((x-self.mu) @ self.invC), axis=-1)


@counter("log_prob")
class ACG(MultivariateNormal):

    def __init__(self, C):
        super().__init__(np.zeros(len(C)), C)

    def log_prob(self, x):
        y = -2 * super().log_prob(x)
        return - np.log(y) * len(self.C) / 2


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
