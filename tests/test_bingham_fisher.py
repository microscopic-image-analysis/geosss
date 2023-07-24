"""
Examples from

A Family of MCMC Methods on Implicitly Defined Manifolds
by Brubaker et al. (2012)

Geodesic Monte Carlo on Embedded Manifolds
by Byrne and Girolami (2013)
"""
import matplotlib.pylab as plt
import numpy as np

import geosss as gs
from geosss.testing import metropolis_hastings, slice_sampling

if __name__ == '__main__':

    np.random.seed(42)

    # Brubaker
    A = np.diag([-1000, -600, -200, 200, 600, 1000])
    b = np.array([100, 0, 0, 0, 0, 0])

    # Girolami
    A = np.diag([-20, -10, 0, 10, 20])

    n_samples = int(2e3)
    stepsize = 0.2

    # showing traces of x_5 analogous to Figure 4 in Byrne and Girolami
    fig, axes = plt.subplots(1, 3, figsize=(9, 2), sharex=True, sharey=True)

    for ax, c1 in zip(axes, [0, 40, 80, 60]):

        b = np.array([c1, 0, 0, 0, 0])
        pdf = gs.BinghamFisher(A, b)
        x = slice_sampling(pdf, n_samples)
        y = metropolis_hastings(pdf, n_samples, stepsize,
                                adapt_stepsize=not True)

        ax.plot(x[:, -1], color=gs.colors[0], alpha=0.7, label='SSS')
        ax.plot(y[:, -1], color=gs.colors[1], alpha=0.7, label='MH')
        ax.set_xlabel('MCMC iteration')
    axes[0].set_ylabel(r'$x_5$')
    ax.set_ylim(-1., 1.)

    fig.tight_layout()

    plt.show()
