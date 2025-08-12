from collections import OrderedDict

import matplotlib.pylab as plt
import numpy as np

import geosss as gs
from geosss.rand import sample_vMF

n_steps = 10
n_samples = 10_000
burnin = 0.1  # percentage no. of samples

# von Mises Fisher distribution as target
d = 10
kappa = (1.0, 10.0, 100.0)[1]
target = gs.VonMisesFisher(kappa * gs.sample_sphere(d - 1))
initial = gs.radial_projection(target.mu)

# samplers
samplers = OrderedDict()
samplers["hmc"] = gs.SphericalHMC(
    target, np.copy(initial), stepsize=1e-3, n_steps=n_steps
)
samplers["sss"] = (
    gs.RejectionSphericalSliceSampler(target, initial),
    gs.ShrinkageSphericalSliceSampler(target, initial),
)[0]

# run samplers
samples = OrderedDict()
for name, sampler in samplers.items():
    with gs.take_time(f"{sampler.__class__.__name__}"):
        samples[name] = sampler.sample(n_samples, burnin)

samples["Wood"] = np.array([sample_vMF(target) for _ in range(n_samples)])

# show samples
hist_kw = dict(density=True, alpha=0.7, color=gs.colors[2])

# HMC
plt.rc("font", size=8)
fig, axes = plt.subplots(2, target.d // 2, figsize=(10, 3), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    hx, bins = np.histogram(samples["Wood"][:, i], bins=30, density=True)
    ax.plot(
        0.5 * (bins[1:] + bins[:-1]),
        hx,
        lw=1,
        alpha=0.99,
        label="Wood",
        color=gs.colors[3],
    )
    ax.hist(samples["hmc"][:, i], bins=bins, label="spherical HMC", **hist_kw)
    ax.set_xlabel(f"$x_{{{i + 1}}}$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks(np.linspace(-1.0, 1.0, 5))
axes[0, 0].legend()
fig.tight_layout()

# Slice sampler
plt.rc("font", size=8)
fig, axes = plt.subplots(2, target.d // 2, figsize=(10, 3), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    hx, bins = np.histogram(samples["Wood"][:, i], bins=30, density=True)
    ax.plot(
        0.5 * (bins[1:] + bins[:-1]),
        hx,
        lw=1,
        alpha=0.99,
        label="Wood",
        color=gs.colors[3],
    )
    ax.hist(samples["sss"][:, i], bins=bins, label="SSS", **hist_kw)
    ax.set_xlabel(f"$x_{{{i + 1}}}$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks(np.linspace(-1.0, 1.0, 5))
axes[0, 0].legend()
fig.tight_layout()

# autocorrelation functions
fig, axes = plt.subplots(2, target.d // 2, figsize=(8, 4), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    acf = gs.acf(samples["hmc"][:, i], 100)
    ax.plot(acf, color=gs.colors[1], lw=2, alpha=0.7, label="HMC")
    acf = gs.acf(samples["sss"][:, i], 100)
    ax.plot(acf, color=gs.colors[2], lw=2, alpha=0.7, label="SSS")
    ax.axhline(0.0, ls="--", color=gs.colors[0], lw=2, alpha=0.8)
    ax.set_xlabel(f"$x_{{{i + 1}}}$")
ax.legend()
ax.set_xlim(0.0, 50.0)
fig.tight_layout()
plt.show()
