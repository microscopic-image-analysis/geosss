import matplotlib.pyplot as plt
import numpy as np
from arviz import autocorr

import geosss as gs


class KacWalk:
    """KacWalk

    Generalized KacWalk.
    """

    def __init__(self, start, prob_angle=None):
        if prob_angle is None:

            def prob_angle():
                return np.pi / 2

        self.prob_angle = prob_angle
        if isinstance(start, int):
            start = gs.sphere.sample_sphere(start)
        self.state = start

    def __next__(self):
        v = self.state
        u = gs.sphere.sample_subsphere(v)
        theta = self.prob_angle()

        self.state = np.cos(theta) * v + np.sin(theta) * u

        return self.state


if __name__ == "__main__":
    dim = 3
    n_samples = 100_000

    walk = (
        KacWalk(dim - 1),
        KacWalk(dim - 1, lambda: np.random.uniform(0.0, 2 * np.pi)),
        KacWalk(dim - 1, lambda: np.pi * 2e-1),
        KacWalk(dim - 1, lambda: np.pi * (1 - 1e-1)),
    )[-1]

    samples = np.array([next(walk) for _ in range(n_samples)])
    phi, theta = gs.sphere.cartesian2spherical(samples)

    plt.close("all")

    hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
    plot_kw = dict(lw=3, alpha=0.9, color=gs.colors[3])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, sharex=True)
    axes[0].hist(phi, **hist_kw)
    axes[1].hist(theta, **hist_kw)
    fig.tight_layout()

    eps = 1e-3
    x = np.linspace(-1.0 + eps, 1.0 - eps, 1000)
    p = np.power(1 - x**2, 0.5 * (walk.state.size - 3))
    p /= p.sum() * (x[1] - x[0])

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.hist(samples[:, i], **hist_kw)
        ax.plot(x, p, **plot_kw)
        ax.set_xlabel(f"$x_{{{i + 1}}}$")
    fig.tight_layout()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        acf = autocorr(samples[:, i])[:20]
        ax.plot(acf, color=gs.colors[1], lw=3, alpha=0.7)
        ax.axhline(0.0, ls="--", color=gs.colors[0], lw=2, alpha=0.8)
        ax.set_xlabel(f"$x_{{{i + 1}}}$")
    fig.tight_layout()

    plt.show()
