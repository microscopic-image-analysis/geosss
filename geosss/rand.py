"""
Special-purpose random number generators for the von Mises-Fisher and the Bingham
distribution.
"""

import numpy as np
from scipy.optimize import brentq

from geosss import distributions as dist
from geosss import sphere


def rotate_north(u):
    """
    Find rotation of u onto north pole [0 ... 0 1].
    """
    v = np.zeros_like(u)
    v[-1] = 1

    U, _, V = np.linalg.svd(np.multiply.outer(v, u))

    R = U @ V

    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1
        R = U @ V

    return R


def sample_vMF(pdf, size=1):
    """
    Generates a random sample from the von Mises-Fisher distribution using the
    algorithm proposed by Wood (1994).
    """
    assert isinstance(pdf, dist.VonMisesFisher)

    if size > 1:
        return np.array([sample_vMF(pdf) for _ in range(int(size))])

    p = pdf.d - 1
    kappa = np.linalg.norm(pdf.mu)

    if np.isclose(kappa, 0.0):
        return sphere.sample_sphere(p)

    u = pdf.mu / kappa

    b0 = (-2 * kappa + (4 * kappa**2 + p**2) ** 0.5) / p
    x0 = (1 - b0) / (1 + b0)
    n = kappa * x0 + p * np.log(1 - x0**2)

    while True:
        Z = np.random.beta(0.5 * p, 0.5 * p)
        U = np.log(np.random.rand())
        W = (1 - (1 + b0) * Z) / (1 - (1 - b0) * Z)

        if (kappa * W + p * np.log(1 - x0 * W) - n) >= U:
            break

    # sample from d-2 sphere
    v = sphere.sample_sphere(p - 1)
    x = np.append((1 - W**2) ** 0.5 * v, W)
    R = rotate_north(u).T

    return R @ x


def sample_bingham_2d(pdf, n_samples=1):
    """Sample from 2D Bingham distribution.

    Sample from the 2D Bingham distribution by transforming it to a zero-
    centered von Mises distribution.
    """
    if not (isinstance(pdf, dist.Bingham) and pdf.d == 2):
        raise ValueError("expected 2D Bingham distribution")

    u = np.random.choice([0, 1], size=int(n_samples))
    kappa = 0.5 * (pdf.v[0] - pdf.v[1])
    phi = np.random.vonmises(0.0, kappa, int(n_samples))

    # von Mises is defined over [-pi, pi), we want phi to be defined over
    # [0, 2pi) therefore we need to shift it by pi
    # theta = 0.5 * np.squeeze(phi + (2 * u + 1) * np.pi)
    theta = 0.5 * np.squeeze(phi + 2 * u * np.pi)

    # Cartesian coordinates
    x = sphere.polar2cartesian(theta)

    return x @ pdf.U.T


def sample_bingham_3d(pdf, n_samples=1):
    """Sample from the 3D Bingham distribution by using a Gibbs sampler."""
    if not isinstance(pdf, dist.Bingham) or pdf.d != 3:
        raise ValueError("3D Bingham expected")

    v = pdf.v

    # initialization
    x = np.random.uniform(-1.0, 1)
    theta = np.arccos(x)

    samples = []
    while len(samples) < n_samples:
        # sample azimuthal angle phi
        kappa = np.sin(theta) ** 2 * 0.5 * (v[0] - v[1])
        u = np.random.choice([0, 1])
        # TODO: phi = 0.5 * (np.random.vonmises(0., kappa) + (2*u + 1) * np.pi)
        phi = 0.5 * (np.random.vonmises(0.0, kappa) + 2 * u * np.pi)

        # sample polar angle theta
        kappa = v[2] + (v[1] - v[0]) * np.cos(phi) ** 2 - v[1]
        C = 0.0 if kappa < 0 else kappa

        # rejection sampling of x=cos(theta)
        while True:
            x = np.random.uniform(-1.0, 1.0)
            u = np.random.rand()
            if np.log(u) < kappa * x**2 - C:
                theta = np.arccos(x)
                break

        samples.append((phi, theta))

    # map to cartesian coordinates
    x = sphere.spherical2cartesian(*np.transpose(samples))

    return x @ pdf.U.T


def bfind(vals):
    """
    `vals` are the eigenvalues of the precision matrix.
    """

    def fb(b):
        return 1 - np.sum(1 / (b + 2 * vals))

    if np.allclose(vals, 0.0):
        return len(vals)
    return brentq(fb, 1, len(vals))


def sample_bingham(A, n_samples, n_iter=1000, return_efficiency=False):
    """
    Implementation of rejection sampling algorithm by Kent, Ganeiber and
    Mardia.
    """
    if A.ndim == 1:
        v = -A
    elif A.ndim == 2:
        v, U = np.linalg.eigh(-A)

    v -= np.min(v)
    b = bfind(v)
    log_M = -(len(A) - b) / 2 + (len(A) / 2) * np.log(len(A) / b)

    n = n_samples
    efficiency = []
    samples = []
    for _ in range(n_iter):
        # sample from angular central Gaussian (ACG) proposal
        x = np.random.randn(n, len(A)) / np.sqrt(1 + 2 * v / b)
        x /= np.linalg.norm(x, axis=-1)[:, None]

        # evaluate unnormalized ACG, Bingham, and bound
        u = np.sum(np.square(x) * v, axis=-1)
        log_acg = -(len(A) / 2) * np.log(1 + 2 * u / b)
        log_bing = -u

        # acceptance probability
        log_prob = log_bing - log_acg - log_M

        # accept / reject
        mask = np.log(np.random.rand(n)) < log_prob
        n -= np.sum(mask)
        samples.append(x[mask])
        efficiency.append(np.mean(mask))
        if n <= 0:
            break

    samples = np.concatenate(samples, 0)[:n_samples]
    if A.ndim == 2:
        samples = samples @ U.T

    if False:
        print("mean efficiency:", np.mean(efficiency))

    if return_efficiency:
        return samples, np.mean(efficiency)

    return samples
