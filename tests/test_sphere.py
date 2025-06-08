import matplotlib.pylab as plt
import numpy as np

import geosss as gs

hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
plot_kw = dict(lw=3, alpha=0.9, color=gs.colors[3])

N = 10_000

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)

for ax1, ax2, d in zip(*axes, [2, 10, 50, 200]):
    dist = np.array(
        [
            gs.distance(gs.sample_sphere(d - 1), gs.sample_sphere(d - 1))
            for _ in range(N)
        ]
    )

    x_1 = np.sqrt(np.random.beta(0.5, 0.5 * (d - 1), size=N))
    x_1 *= np.random.choice([-1, 1], size=N)
    dist2 = np.arccos(x_1)

    # theory
    theta = np.linspace(0.0, np.pi, 1000)
    p = np.power(np.sin(theta), d - 2)
    p /= (theta[1] - theta[0]) * p.sum()

    ax1.hist(dist, **hist_kw)
    ax1.plot(theta, p, **plot_kw)
    ax2.hist(dist2, **hist_kw)
    ax2.plot(theta, p, **plot_kw)

    ax1.set_title(f"$n={d}$")

fig.tight_layout()

d = 100
u = gs.sample_sphere(d - 1)
v = gs.sample_sphere(d - 1)
x = gs.sample_sphere(d - 1)

v = x
u = gs.spherical_projection(u, v)
R = gs.givens(u, v, x)

assert np.allclose(R(0.0), x)
assert np.allclose(R(2 * np.pi), x)
assert np.allclose(R(np.pi), -x)
assert np.isclose(R(0.5 * np.pi) @ x, 0.0)

plt.show()
