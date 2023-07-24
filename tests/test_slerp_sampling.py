# testing the rejection sampling procedure for a 3d bingham distribution

import matplotlib.pyplot as plt
import numpy as np
from csb.statistics import autocorrelation
from scipy.special import logsumexp

import geosss as gs
import geosss.testing as testing

pdf = gs.random_bingham(d=3, vmax=10, vmin=0., eigensystem=not True)

# importance sampling
X = gs.sample_sphere(2, 1000000)
p = pdf.log_prob(X)
p -= logsumexp(p)
np.exp(p, p)

# compute marginal distributions
angles = gs.cartesian2spherical(X)
p_spherical = []
p_cartesians = []
for a in angles:
    hist, bins = np.histogram(a, bins=70, weights=p, density=True)[:2]
    p_spherical.append((0.5 * (bins[:-1] + bins[1:]), hist))
for z in X.T:
    hist, bins = np.histogram(z, bins=70, weights=p, density=True)[:2]
    p_cartesians.append((0.5 * (bins[:-1] + bins[1:]), hist))

# run handcrafted sampler
n_samples = 10000
burnin = int(0.1 * n_samples)
x = gs.sample_bingham_3d(pdf, n_samples)

# Metropolis-Hastings
y = testing.metropolis_hastings(pdf, burnin + n_samples, stepsize=0.5,
                                adapt_stepsize=not True)[burnin::]

# slerp and bracketing sampling
z = testing.slice_sampling3(pdf, burnin + n_samples)[burnin:]

# run method by Kent, Ganeiber & Mardia
w = gs.sample_bingham(pdf.A, n_samples)

# show marginal distributions of Cartesian coordinates
hist_kw = dict(bins=50, density=True, alpha=0.7, color=gs.colors[2])
plot_kw = dict(lw=1, alpha=0.9, color=gs.colors[3])
plt.rc('font', size=10)
fig, axes = plt.subplots(4, 3, figsize=(7, 7), sharex=True, sharey=True)
for i, (ax, ax2, ax3, ax4) in enumerate(zip(*axes)):
    ax.hist(x[:, i], **hist_kw)
    ax.plot(*p_cartesians[i], **plot_kw)
    ax2.hist(y[:, i], **hist_kw)
    ax2.plot(*p_cartesians[i], **plot_kw)
    ax3.hist(z[:, i], **hist_kw)
    ax3.plot(*p_cartesians[i], **plot_kw)
    ax4.hist(w[:, i], **hist_kw)
    ax4.plot(*p_cartesians[i], **plot_kw)
    ax4.set_xlabel(f'$x_{i+1}$')
axes[0, 1].set_title('Gibbs sampler')
axes[1, 1].set_title('Metropolis-Hastings')
axes[2, 1].set_title('Spherical slice sampling using slerp/bracketing')
axes[3, 1].set_title('Kent\'s algorithm')
fig.tight_layout()

# show marginal distributions of spherical coordinates
sphere = gs.sphere
a = sphere.cartesian2spherical(x)
b = sphere.cartesian2spherical(y)
c = sphere.cartesian2spherical(z)
d = sphere.cartesian2spherical(w)

labels = (r'$\varphi$', r'$\theta$')

fig, axes = plt.subplots(4, 2, figsize=(7, 7), sharex=True, sharey=True)
for i, (ax1, ax2, ax3, ax4) in enumerate(zip(*axes)):
    for ax, xx in zip([ax1, ax2, ax3, ax4], [a, b, c, d]):
        ax.hist(xx[i], **hist_kw)
        ax.plot(*p_spherical[i], **plot_kw)
    ax4.set_xlabel(labels[i])
axes[0, 0].set_title('Gibbs sampler')
axes[1, 0].set_title('Metropolis-Hastings')
axes[2, 0].set_title('Spherical slice sampling using slerp/bracketing')
axes[3, 0].set_title('Kent\'s algorithm')
fig.tight_layout()

# show 3D samples as scatter plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 10))
ax.scatter(*x.T, s=1, alpha=0.92, color='r')
ax.scatter(*w.T, s=1, alpha=0.92, color='b')

# show autocorrelation functions
fig, axes = plt.subplots(4, 3, figsize=(7, 7), sharex=True, sharey=True)
for i, (ax1, ax2, ax3, ax4) in enumerate(zip(*axes)):
    for ax, xx in zip([ax1, ax2, ax3, ax4], [x, y, z, w]):
        ax.plot(autocorrelation(xx[:, i], 50), color='k', lw=3, alpha=0.7)
        ax.axhline(0., ls='--', color='r', lw=2, alpha=0.8)
    ax4.set_xlabel(f'$x_{i+1}$')
axes[0, 1].set_title('Gibbs sampler')
axes[1, 1].set_title('Metropolis-Hastings')
axes[2, 1].set_title('Spherical slice sampling using slerp/bracketing')
axes[3, 1].set_title('Kent\'s algorithm')
for ax in axes[:, 0]:
    ax.set_ylabel('autocorrelation')
fig.tight_layout()
plt.show()

# assign samples to closest mode
dx_mode = [sphere.distance(pdf.mode, x), sphere.distance(-pdf.mode, x)]
dy_mode = [sphere.distance(pdf.mode, y), sphere.distance(-pdf.mode, y)]
dz_mode = [sphere.distance(pdf.mode, z), sphere.distance(-pdf.mode, z)]
dw_mode = [sphere.distance(pdf.mode, w), sphere.distance(-pdf.mode, w)]

cx = (dx_mode[0] < dx_mode[1]).astype('d')
cy = (dy_mode[0] < dy_mode[1]).astype('d')
cz = (dz_mode[0] < dz_mode[1]).astype('d')
cw = (dw_mode[0] < dw_mode[1]).astype('d')

print(('populations: {0:.1%} (Gibbs) {1:.1%} (MH) {2:.1%} (SSS) '
       '{3:.1%} (Kent)').format(
    np.mean(cx), np.mean(cy), np.mean(cz), np.mean(cw))
)

print(('average hopping frequency: {0:.2e} (Gibbs) {1:.2e} (MH) '
       '{2:.1%} (SSS) {3:.1%} (Kent)').format(
    np.mean(np.fabs(np.diff(cx))), np.mean(np.fabs(np.diff(cy))),
    np.mean(np.fabs(np.diff(cz))), np.mean(np.fabs(np.diff(cw)))))
