"""
Some utility functions for sampling from the sphere. 
"""

import numpy as np

# projections


def radial_projection(x):
    """
    Radial projection of (d+1)-dimensional point(s) to d-sphere.
    """
    norm = np.linalg.norm(x, axis=-1) + 1e-100
    if x.ndim == 1:
        return x / norm
    else:
        return x / norm[:, None]


def orthogonal_projection(x, y):
    """
    Map point(s) `x` into orthogonal complement of point `y`.
    """
    normal = radial_projection(y)
    return x - (x @ normal) * normal


def spherical_projection(x, v):
    """
    Map point(s) `x` into the great subsphere with pole `v`.
    """
    return radial_projection(orthogonal_projection(x, v))


# sampling


def sample_sphere(d=2, size=None, seed=None):
    """
    Draw a random point from d-sphere by drawing a (d+1)-dimensional point from the
    standard normal distribution and mapping it to d-sphere.
    """
    rng = np.random.default_rng(seed)
    x = (
        rng.standard_normal(d + 1)
        if size is None
        else rng.standard_normal((size, d + 1))
    )
    return radial_projection(x)


def sample_subsphere(v, seed=None):
    """
    Sample uniformly from the great subsphere with pole `v`.
    """
    rng = np.random.default_rng(seed)
    return spherical_projection(rng.standard_normal(len(v)), v)


# distances


def distance(x, y):
    """
    Great circle distance (assuming x, y are on the sphere).
    """
    return np.arccos(np.clip(np.sum(x * y, axis=-1), -1, 1))


# transformation between Cartesian and polar coordinates


def cartesian2polar(x):
    """Map points on the unit sphere to polar coordinates."""
    return np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)


def polar2cartesian(theta):
    return np.transpose([np.cos(theta), np.sin(theta)])


def spherical2cartesian(phi, theta):
    return np.transpose(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
    )


def cartesian2spherical(x):
    return cartesian2polar(x), np.mod(np.arccos(x[:, 2]), np.pi)


def sample_marginal(d, size=None, seed=None):
    rng = np.random.default_rng(seed)
    s = rng.beta(0.5, 0.5 * (d - 1), size=size)
    t = np.sqrt(s) * rng.choice([-1, 1], size=size)
    return t


def wrap(x, u, v):
    """See Mardia and Jupp for 'wrapping approach'."""
    theta = np.linalg.norm(x)
    return np.sin(theta) * u + np.cos(theta) * v


def slerp(u, v):
    theta = np.arccos(u @ v)

    def interpolation(phi, u=u, v=v, theta=theta):
        return (np.sin(phi) * v + np.sin(theta - phi) * u) / np.sin(theta)

    return interpolation


def givens(u, v, x):
    """
    Returns function that rotates `x` in the plane spanned by `u`, `v` by an
    angle that will be the argument of the function.
    """

    def rotate(theta, u=u, v=v, x=x):
        ux = u @ x
        vx = v @ x
        return (
            x
            + (np.cos(theta) - 1) * (ux * u + vx * v)
            + np.sin(theta) * (ux * v - vx * u)
        )

    return rotate
