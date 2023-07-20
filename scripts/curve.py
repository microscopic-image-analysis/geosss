import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as opt
from csb.io import dump, load
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from tsp_solver.greedy import solve_tsp

import geosss as gs


def distance_slerp(x, a, b):
    """Compute the shortest distance between a point `x` on the sphere and the slerp
    connecting two points on the sphere `a`, `b`.

    Parameters
    ----------
    x : point on sphere
        Query point whose distance from the slerp will be determined
    a : point on sphere
        Start of slerp
    b : point on sphere
        End of slerp

    Returns
    -------
    * Shorest geodesic distance from point to slerp
    * Closest point on slerp
    """
    theta = gs.sphere.distance(a, b)
    t = np.arctan2(b @ x - a @ x * np.cos(theta), a @ x * np.sin(theta))
    t = np.clip(t, 0.0, theta)
    y = (np.sin(theta - t) * a + np.sin(t) * b) / np.sin(theta)
    return gs.sphere.distance(x, y), y


def saff_sphere(N: int = 1000) -> np.ndarray:
    """Uniformly distribute points on the 2-sphere using Saff's algorithm."""
    h = np.linspace(-1, 1, N)
    theta = np.arccos(h)
    incr = 3.6 / np.sqrt(N * (1 - h[1:-1] ** 2))
    phi = np.add.accumulate(np.append(0, incr))
    phi = np.append(phi, 0.)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return np.transpose([x, y, z])


class SphericalCurve:
    def __init__(self, knots):
        self.knots = knots

    @classmethod
    def random_curve(cls, n_pts=10):
        """Generate random points on 2-sphere and order them by finding the shortest
        path that runs through all points. These points will be interpolated by a curve
        on the sphere.
        """
        x = gs.sphere.sample_sphere(2, n_pts)
        return cls(x[np.array(solve_tsp(cdist(x, x)))])


class SphericalSpline(SphericalCurve):

    def __init__(self, knots):
        super().__init__(knots)
        self.spline, _ = interpolate.splprep(knots.T, s=2)

    def __call__(self, t):
        return gs.sphere.map_to_sphere(np.transpose(interpolate.splev(t, self.spline)))

    def find_nearest(self, x, n_pts=100):
        def func(t, x=x, curve=self):
            return np.linalg.norm(curve(t) - x)
        t = np.linspace(0.0, 1.0, n_pts)
        t0 = t[np.linalg.norm(self(t) - x, axis=1).argmin()]
        if np.isclose(t0, 0.0) or np.isclose(t0, 1.0):
            bracket = (0.0, 1.0)
        else:
            bracket = (0.0, t0, 1.0)
        t = opt.minimize_scalar(func, bracket=bracket).x
        return self(t)


class SlerpCurve(SphericalCurve):

    def __init__(self, knots):
        super().__init__(knots)
        self.theta = np.array([gs.sphere.distance(a, b)
                              for a, b in zip(knots, knots[1:])])
        self.theta = np.append(0., self.theta)
        self.bins = np.add.accumulate(self.theta)
        self.bins /= self.bins[-1]
        self.widths = np.diff(self.bins)

    def __call__(self, t):
        s = self.bins
        i = np.clip(np.digitize(t, s, right=True), 1, len(self.knots) - 1)
        theta = self.theta[i]
        a = np.sin(theta * (s[i] - t) / (s[i] - s[i-1])) / np.sin(theta)
        b = np.sin(theta * (t - s[i-1]) / (s[i] - s[i-1])) / np.sin(theta)
        return a[:, None] * self.knots[i-1] + b[:, None] * self.knots[i]

    def find_nearest(self, x):
        distances = []
        points = []
        for a, b in zip(self.knots, self.knots[1:]):
            d, y = distance_slerp(x, a, b)
            distances.append(d)
            points.append(y)
        return points[np.argmin(distances)]


@gs.counter(["log_prob", "gradient"])
class CurvedVonMisesFisher(gs.Distribution):
    def __init__(self, curve: SphericalCurve, kappa: float = 100.0):
        self.curve = curve
        self.kappa = kappa

    def log_prob(self, x):
        if x.ndim == 1:
            return self.kappa * (x @ self.curve.find_nearest(x))
        return np.array(list(map(self.log_prob, x)))

    def gradient(self, x):
        return self.kappa * self.curve.find_nearest(x)


def test_gradient(pdf):
    x = gs.sphere.sample_sphere()
    a = pdf.gradient(x)
    b = opt.approx_fprime(x, pdf.log_prob, 1e-7)
    print(a)
    print(b)
    assert np.allclose(a, b)

    # optional check (internally compares using `appox_fprime`)
    # error should be low!
    err = opt.check_grad(pdf.log_prob, pdf.gradient, x, seed=342)
    print(f"error to check correctness of gradient:, {err}")


def launch_samplers(savedir, kappa, pdf, tester, methods):
    """ just an interface to load or run samplers """

    # load saved samples
    savepath_samples = f"{savedir}/curve_samples_kappa{int(kappa)}.pkl"
    savepath_logprob = f"{savedir}/curve_logprob_kappa{int(kappa)}.pkl"
    try:
        samples = load(savepath_samples)
        print(f"Loading file {savepath_samples}")

        logprob = load(savepath_logprob)
        print(f"Loading file {savepath_logprob}")

    # run samplers
    except FileNotFoundError:
        print("File not found, starting samplers..")

        def start_samplers():
            """ just a util function to pass the output of 
            this in log file
            """

            # run samplers
            samples = {}
            for method in methods:
                with gs.take_time(method):
                    samples[method] = tester.run(method)

                    print("\n---------------------------"
                          f"Starting the sampler {method}"
                          "---------------------------")

                    # no. of gradient and log_prob calls
                    print(f"gradient calls for {method}:",
                          pdf.gradient.num_calls)
                    print(f"logprob calls for {method}:",
                          pdf.log_prob.num_calls)

                    # counter for rejected samples
                    if method == 'sss-reject':
                        print(f"Rejected samples for {method}:"
                              f"{tester.rsss.n_reject}")

                    if method == 'sss-shrink':
                        print(f"Rejected samples for {method}:"
                              f"{tester.ssss.n_reject}")

            logprob = {}
            for method in methods:
                logprob[method] = pdf.log_prob(samples[method])

            # save the runs
            dump(samples, savepath_samples)
            dump(logprob, savepath_logprob)

            return samples, logprob

        # save the print output to a log file
        with open(f'{savedir}/curve_kappa{int(kappa)}_log.txt', 'w') as f:
            with redirect_stdout(f):
                samples, logprob = start_samplers()

    return samples, logprob


def visualize_samples(samples: dict, methods: tuple):
    """ visualize samples on a 3d sphere """
    phi, theta = np.mgrid[0:np.pi:20j, 0:2*np.pi:30j]
    euler = (np.sin(phi) * np.cos(theta),
             np.sin(phi) * np.sin(theta),
             np.cos(phi))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), subplot_kw=dict(projection='3d'),
                             sharex=True, sharey=True)

    for ax, method in zip(axes.flat, methods):
        ax.computed_zorder = False
        ax.plot_wireframe(*euler, color='green', alpha=0.06, zorder=1)
        ax.plot_surface(*euler, cmap='viridis', alpha=0.07, zorder=2)
        x = samples[method][:int(1e4)]
        ax.set_title(algos[method])
        ax.plot(*curve(t).T, color='r', alpha=0.9, lw=3, zorder=3)
        ax.scatter(*x.T, c='k', s=1, alpha=0.08, zorder=4)
        ax.set_aspect("equal")
        ax.view_init(-140, 20)
    fig.tight_layout()

    return fig


if __name__ == '__main__':

    t = np.linspace(0, 1, 1_000)   # points on curve
    kappa = 300.0                  # concentration parameter
    fix_curve = True               # fix curve (target)
    reprod_switch = True           # seeds samplers for reproducibility
    n_samples = int(1e5)           # number of samples per sampler
    burnin = int(0.1 * n_samples)  # burn-in
    savefig = True                 # save the plots

    # directory to save results
    savedir = f"results/vMF_curve_kappa{int(kappa)}"
    os.makedirs(savedir, exist_ok=True)

    # define curve on the sphere
    if fix_curve:
        knots = np.array([[-0.25882694,  0.95006168,  0.17433133],
                          [0.14557335,  0.61236727,  0.77705516],
                          [-0.7973001, -0.25170369,  0.54859622],
                          [0.03172733, -0.71944851,  0.69382074],
                          [0.56217797, -0.29453368,  0.77279094],
                          [0.80883044,  0.1316755,  0.57310983],
                          [0.98981463,  0.03039439, -0.13907979],
                          [0.81592815,  0.04723609, -0.57622045],
                          [0.36888235,  0.400026, -0.83899047],
                          [-0.6770828,  0.05213374, -0.73405787]])
        curve = SlerpCurve(knots)
    else:
        curve = SlerpCurve.random_curve(10)

    pdf = CurvedVonMisesFisher(curve, kappa)

    # eval density
    x = saff_sphere(5000)
    log_p = pdf.log_prob(x)
    p = np.exp(log_p - logsumexp(log_p))

    # show curve on the sphere
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='3d'))
    ax.set_box_aspect((1, 1, 1))
    ax.scatter(*x.T, c=p, s=10, alpha=0.15)
    ax.plot(*curve(t).T, color='k', alpha=1.0)
    ax.scatter(*curve(t).T, c=t, s=1)
    fig.tight_layout()

    # initial state fixed and samplers seeded for reproducibility
    initial = np.array([0.65656515, -0.63315859, -0.40991755])
    seed = 6756 if reprod_switch else None

    # `tester` instances samplers
    launcher = gs.SamplerLauncher(pdf, initial, n_samples, burnin, seed)
    methods = ('sss-reject', 'sss-shrink', 'rwmh', 'hmc')
    algos = {'sss-reject': 'geoSSS (reject)', 'sss-shrink': 'geoSSS (shrink)',
             'rwmh': 'RWMH', 'hmc': 'HMC'}

    # load samples by running or loading from memory
    samples, logprob = launch_samplers(savedir, kappa, pdf, launcher, methods)

    # plot samples on a 3d sphere
    fig = visualize_samples(samples, methods)
    if savefig:
        fig.savefig(f'{savedir}/curve_samples_kappa{int(kappa)}.pdf',
                    bbox_inches='tight', transparent=True)

    # generate figures
    fs = 16
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for d, ax in enumerate(axes):
        ax.set_title(rf"$x_{d+1}$", fontsize=20)
        for method in methods:
            ac = gs.acf(samples[method][:, d], 250)
            ax.plot(ac, alpha=0.7, lw=3, label=algos[method])
        ax.axhline(0., ls='--', color='k', alpha=0.7)
        ax.set_xlabel(r'lag', fontsize=fs)
    axes[0].set_ylabel('ACF', fontsize=fs)
    ax.legend(fontsize=fs)
    fig.tight_layout()
    if savefig:
        fig.savefig(f'{savedir}/curve_acf_kappa{int(kappa)}.pdf',
                    bbox_inches='tight', transparent=True)

    # geodesic distance
    fig, axes = plt.subplots(1, len(methods), figsize=(len(methods) * 3, 3),
                             sharex=True, sharey=True)
    bins = 100
    for ax, method in zip(axes, methods):
        ax.set_title(algos[method], fontsize=fs)
        # distance between successive samples
        x = samples[method]
        d = gs.distance(x[:-1], x[1:])
        print('average great circle distance of successive samples: '
              f'{np.mean(d):.2f} ({method})')
        bins = ax.hist(d, bins=bins, density=True, alpha=0.3,
                       color='k', histtype='stepfilled')[1]
        ax.set_xlabel(r'$\delta(x_{n+1}, x_n)$', fontsize=fs)
        ax.set_xticks(np.linspace(0., np.pi, 3))
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        ax.semilogy()
    fig.tight_layout()
    if savefig:
        fig.savefig(f'{savedir}/curve_dist_kappa{int(kappa)}.pdf',
                    bbox_inches='tight', transparent=True)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
    for ax, method in zip(axes[0], methods):
        ax.set_title(algos[method], fontsize=fs)
    for d in range(3):
        for ax, method in zip(axes[d], methods):
            ac = gs.acf(samples[method][:, d], 1000)
            ax.plot(ac, alpha=0.7, color='k', lw=3)
            ax.axhline(0., ls='--', color='r', alpha=0.7)
    for ax in axes[-1]:
        ax.set_xlabel(r'Lag', fontsize=fs)
    for d, ax in enumerate(axes[:, 0], 1):
        ax.set_ylabel(fr'ACF $x_{d}$', fontsize=fs)
    ax.set_xlim(-5, 250)
    fig.tight_layout()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ac = gs.acf(logprob[method], 1000)
        ax.plot(ac, color='k', alpha=1.0, lw=3)
    ax.set_xlim(-5, 200)
    fig.tight_layout()

    # compare histograms

    x = saff_sphere(1500)
    log_p = pdf.log_prob(x)
    p = np.exp(log_p - logsumexp(log_p))

    tree = cKDTree(x)

    kl = []
    for method in methods:
        d, i = tree.query(samples[method], k=1)
        j, c = np.unique(i, return_counts=True)
        q = np.zeros_like(p)
        q[j] = c = c / c.sum()
        kl.append(np.sum(p * np.log(p) - p * np.log(q + p.min())))
        print(method, kl[-1])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    ax = axes
    # ax.set_title("KL divergence between target and sampled distribution")
    ax.set_ylabel("KL divergence")
    ax.bar(list(map(algos.get, methods)), kl, color='k', alpha=0.3)
    plt.xticks(rotation=30)
    fig.tight_layout()
    if savefig:
        fig.savefig(f"{savedir}/curve_kl_kappa{int(kappa)}.pdf",
                    bbox_inches='tight', transparent=True)

    x = saff_sphere(100_000)
    log_p = pdf.log_prob(x)
    p = np.exp(log_p - logsumexp(log_p))

    bins = 50
    plt.rc("font", size=fs)
    fig, rows = plt.subplots(3, len(methods), figsize=(12, 10),
                             sharex=True, sharey=True)
    for i, axes in enumerate(rows):
        vals = x[:, i]
        ref = list(np.histogram(vals, weights=p, bins=bins, density=True))
        ref[1] = 0.5 * (ref[1][1:] + ref[1][:-1])
        for ax, method in zip(axes, methods):
            bins = ax.hist(samples[method][burnin:, i], bins=bins, density=True,
                           alpha=0.3, color='k', histtype='stepfilled')[1]
            ax.plot(*ref[::-1], color='r', lw=1, ls='--')
            ax.set_xlabel(fr'$e_{i}^Tx_n$', fontsize=fs)
    for ax, method in zip(rows[0], methods):
        ax.set_title(algos[method], fontsize=fs)
    fig.tight_layout()
    if savefig:
        fig.savefig(f"{savedir}/curve_hist_kappa{int(kappa)}.pdf",
                    bbox_inches='tight', transparent=True)
