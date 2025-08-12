import matplotlib.pyplot as plt
import numpy as np

import geosss as gs
import tests.testing as testing
from geosss.rand import sample_bingham_2d

pdf = gs.random_bingham(d=2, vmax=5, eigensystem=not True)
theta = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
p = np.exp(pdf.log_prob(gs.polar2cartesian(theta)))
p /= (theta[1] - theta[0]) * p.sum()

x = sample_bingham_2d(pdf, 10000)
y = testing.metropolis_hastings(pdf, 50000)[5000::10]

plt.close("all")
hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
plot_kw = dict(lw=3, alpha=0.9, color=gs.colors[3])
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(theta, p, **plot_kw)
axes[0].hist(gs.sphere.cartesian2polar(x), **hist_kw)
axes[1].scatter(*(x + 0.02 * np.random.randn(*x.shape)).T, color="k", alpha=0.01, s=2)
axes[2].scatter(*(y + 0.02 * np.random.randn(*y.shape)).T, color="k", alpha=0.01, s=2)
fig.tight_layout()
plt.show()
