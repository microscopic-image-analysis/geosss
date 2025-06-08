"""
Some tests on the 2D and 3D Bingham distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from arviz import autocorr as autocorrelation
from scipy.special import logsumexp

import geosss as gs
from tests.testing import sample_slice, slice_sampling, spherical_rejection_sampling


def test_rejection_sampling():
    """
    Test if sampling from exp(k * x**2) for x in [-1, 1] works.
    """
    k = np.random.uniform(-10.0, 10.0)
    C = 0.0 if k < 0 else k
    X = []
    while len(X) < 1e4:
        x = np.random.uniform(-1.0, 1.0)
        u = np.log(np.random.rand())
        if u < k * x**2 - C:
            X.append(x)

    x = np.linspace(-1.0, 1.0, 1000)
    p = np.exp(k * x**2)
    p /= p.sum() * (x[1] - x[0])

    hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
    plot_kw = dict(lw=3, alpha=0.9, color=gs.colors[3])
    fig, ax = plt.subplots()
    ax.plot(x, p, **plot_kw)
    ax.hist(X, **hist_kw)
    plt.show()


def test_slice_sampling():
    """
    Test slice sampler on p(x) = exp(k * x**2) for x in [-1, 1]
    """
    k = np.random.uniform(-10.0, 10.0)
    C = 0.0 if k < 0 else k
    bounds = -1.0, 1.0

    def log_prob(x):
        return k * x**2

    X = [np.random.uniform(*bounds)]
    while len(X) < 1e4:
        X.append(sample_slice(log_prob, X[-1], bounds))

    x = np.linspace(*bounds, 1000)
    p = np.exp(k * x**2)
    p /= p.sum() * (x[1] - x[0])

    hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
    plot_kw = dict(lw=3, alpha=0.9, color=gs.colors[3])
    fig, ax = plt.subplots()
    ax.plot(x, p, **plot_kw)
    ax.hist(X, **hist_kw)
    plt.show()


def test_uniform(d=10, method="bracketing"):
    """
    Uniform sampling by using Bingham with A=0.
    """
    # points from Bingham with A=0 should be uniform on the sphere
    pdf = gs.random_bingham(d)
    pdf.A *= 0.0
    if method == "bracketing":
        x = slice_sampling(pdf, 10000)
    else:
        x = spherical_rejection_sampling(pdf, 10000)

    # marginal distribution of x_i on the sphere
    eps = 1e-3
    t = np.linspace(-1.0 + eps, 1.0 - eps, 1000)
    p = np.power(1 - t**2, 0.5 * (d - 3))
    p /= p.sum() * (t[1] - t[0])

    hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
    plot_kw = dict(lw=3, alpha=0.9, color=gs.colors[3])

    fig, axes = plt.subplots(2, pdf.d // 2, figsize=(12, 6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.hist(x[:, i], **hist_kw)
        ax.plot(t, p, **plot_kw)
        ax.set_xlabel(f"$x_{{{i + 1}}}$")
    fig.tight_layout()

    fig, axes = plt.subplots(2, pdf.d // 2, figsize=(12, 6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        acf = autocorrelation(x[:, i])[:10]
        ax.plot(acf, color="k", lw=3, alpha=0.7)
        ax.axhline(0.0, ls="--", color="r", lw=2, alpha=0.8)
        ax.set_xlabel(f"$x_{{{i + 1}}}$")
    fig.tight_layout()
    plt.show()


def test_projection(d=11, N=10_000):
    """
    Check distribution of coefficients `alpha` and `beta` (as defined in the
    paper).
    """
    # generate some `v` that will define the normal vector of the plane
    v = gs.sphere.sample_sphere(d - 1)

    # projector onto plane
    P = np.eye(d) - v[:, None] * v
    l, U = np.linalg.eigh(P)
    U = U[:, 1:]

    assert np.allclose(P, U @ U.T)
    assert np.allclose(l[1:], 1.0)

    # some random points from the standard normal that will be projected to the
    # plane
    x = np.random.randn(N, d)

    # projections
    beta = x @ U
    alpha = x @ v

    assert np.allclose(beta @ U.T, x @ P)

    t = np.linspace(-1, 1, 1000) * 5
    p = np.exp(-0.5 * t**2)
    p /= p.sum() * (t[1] - t[0])

    hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
    plot_kw = dict(lw=3, alpha=0.9, color=gs.colors[3])

    fig, axes = plt.subplots(2, d // 2, figsize=(12, 5), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.hist(beta[:, i], **hist_kw)
        ax.plot(t, p, **plot_kw)
        ax.set_xlabel(r"$\beta_{{{}}}$".format(i + 1))
    fig.tight_layout()
    plt.show()


def show_bingham_3d():
    """Just visualizing 3d bingham on a sphere"""

    # Trying to visualize 3D Bingham
    pdf = gs.random_bingham(d=3, vmax=10, vmin=0.0, eigensystem=not True)

    eps = 1e-3
    n = 100
    x = np.reshape(
        np.mgrid[-1.0 : 1 : 1j * n, -1.0 : 1 : 1j * n, -1.0 : 1 : 1j * n], (3, -1)
    ).T
    r = np.linalg.norm(x, axis=1)
    mask = (1 - eps <= r) & (r <= 1 + eps)
    x = x[mask]

    n = 200
    phi = np.linspace(0.0, 2 * np.pi, n)
    theta = np.arccos(np.linspace(-1.0, 1.0, n))

    phi, theta = np.meshgrid(phi, theta, indexing="ij")

    x = gs.sphere.spherical2cartesian(phi.flatten(), theta.flatten())

    logp = pdf.log_prob(x)
    logp -= logsumexp(logp)

    # scatter plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(10, 10))
    ax.scatter(*x.T, s=100, alpha=0.92, c=np.exp(logp))
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    show_bingham_3d()

    if False:
        test_rejection_sampling()
    if False:
        test_slice_sampling()
    if False:
        test_uniform(method="bracketing")
        test_uniform(method="rejection-sampling")
    if False:
        test_projection()
