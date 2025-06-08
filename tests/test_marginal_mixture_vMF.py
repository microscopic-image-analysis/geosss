import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import iv

import geosss as gs


class MarginalVonMisesFisher(gs.VonMisesFisher):
    def __init__(self, index, mu):
        super().__init__(mu)
        self.index = index

    def prob(self, x):
        d = len(self.mu)
        kappa = np.linalg.norm(self.mu)
        mu = self.mu[self.index] / kappa
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


if __name__ == "__main__":
    d = 10 * 4
    kappa = 100.0
    mu = gs.sample_sphere(d - 1)
    pdf = gs.VonMisesFisher(kappa * mu)
    marginals = [MarginalVonMisesFisher(i, pdf.mu) for i in range(d)]
    x = gs.sample_vMF(pdf, 10_000)
    t = np.linspace(-1.0, 1.0, 1_000)

    fig, axes = plt.subplots(d // 5, 5, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        marginal = marginals[i]
        assert np.isclose(quad(marginal.prob, -1, 1)[0], 1.0)
        ax.hist(x[:, i], bins=50, density=True, alpha=0.25, color="k")
        ax.plot(t, marginal.prob(t), ls="--", color="r")
        ax.axvline(mu[i], ls="--", color="k", alpha=0.5)
    fig.tight_layout()

    plt.show()

if False:
    from scipy.spatial.distance import squareform

    mode_seed = 1234
    kappa = 100.0
    d = 10
    K = 5
    modes = gs.sphere.sample_sphere(d - 1, K, seed=mode_seed)
    angles = squareform(np.degrees(np.arccos(np.clip(modes @ modes.T, -1, 1))))
    print(angles.min(), angles.max())
    pdf = gs.MixtureModel([gs.VonMisesFisher(kappa * mu) for mu in modes])
    marginals = []
    for i in range(d):
        marginal = gs.MixtureModel([MarginalVonMisesFisher(i, p.mu) for p in pdf.pdfs])
        marginal.weights[::] = pdf.weights
        marginals.append(marginal)

    N = np.random.multinomial(10_000, pdf.weights)
    x = np.vstack([gs.sample_vMF(p, n) for p, n in zip(pdf.pdfs, N)])
    np.random.shuffle(x)

    fig, axes = plt.subplots(d // 5, 5, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        marginal = marginals[i]
        assert np.isclose(quad(lambda x: np.exp(marginal.log_prob(x)), -1, 1)[0], 1.0)
        ax.hist(x[:, i], bins=50, density=True, alpha=0.25, color="k")
        ax.plot(t, np.exp(marginal.log_prob(t)), ls="--", color="r")
        ax.axvline(mu[i], ls="--", color="k", alpha=0.5)
    fig.tight_layout()
