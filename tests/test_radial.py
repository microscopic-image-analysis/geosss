"""
Empirically test correctness of the marginal distribution of the radius.
"""
import matplotlib.pylab as plt
import numpy as np
from scipy.special import logsumexp

# random points from standard normal distribution
d = 10
x = np.random.randn(10_000, d)
r = np.linalg.norm(x, axis=-1)

# sample radius from gamma distribution
r2 = np.sqrt(2 * np.random.gamma(d / 2, size=10_000))

# show samples radius
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
_, bins, *_ = axes[0].hist(r, bins=50, density=True, color='k', alpha=0.1)

# theoretical distribution of the radius
s = 0.5 * (bins[1:] + bins[:-1])
p = (d - 1) * np.log(s) - s**2 / 2
p -= logsumexp(p) + np.log(s[1] - s[0])

axes[1].hist(r2, bins=50, density=True, color='k', alpha=0.1)
for ax in axes:
    ax.plot(s, np.exp(p), color='r', lw=3)
fig.tight_layout()

plt.show()
