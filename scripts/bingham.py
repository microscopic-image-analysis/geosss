import os

import arviz as az
import matplotlib.pylab as plt
import numpy as np
from csb.io import dump, load

import geosss as gs


def random_bingham(d=2, vmax=None, vmin=None, eigensystem=False, seed=None):
    """
    Create a random Bingham distribution where
    Parameters
    ----------
    d : integer >= 2
        Dimension of ambient space
    vmax : float or None
        Optional maximum eigenvalue of the precision matrix.
    vmin : float or None
        Optional minimum eigenvalue of the precision matrix.
    eigensystem : bool
        Flag specifying if Bingham distribution has diagonal precision matrix 
        (i.e. we are working in the eigenvasis of A) or not. (Default value:
        False)
    """
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((d, d))
    A = A.T @ A
    v, U = np.linalg.eigh(A)
    if vmin is not None:
        v += vmin - v.min()
    if vmax is not None:
        v *= vmax / v.max()
    if eigensystem:
        U = np.eye(d)
    A = (U * v) @ U.T
    return gs.Bingham(A)


class SamplerLauncher(gs.SamplerLauncher):
    """ Adding the kent method to the sampler interface to generate 
    ground truth samples
    """

    def run_kent(self):
        return gs.sample_bingham(self.pdf.A, self.n_samples)

    def run(self, method):
        return self.run_kent() if method == 'kent' else super().run(method)


def hopping_frequency(samples, pdf):
    """ hopping frequency between the two modes of bingham """
    return np.mean(np.diff(np.sign(samples @ pdf.mode)) != 0.0)


def bingham_ess(runs_samples, pdf, methods, path, return_ess=True):
    """ 
    calculates ess using the arviz library with the default 'bulk' method
    and saves the result. This implementation implements for multidimensional 
    target and estimates ess values per dimension using `n` chains 
    """
    ess_file = f"{path}_ess.pkl.gz"

    # load or calculate ess (and then save)
    try:
        ess = load(ess_file, gzip=True)
        print(f"Loading ESS file {ess_file}")

    except FileNotFoundError:

        # calculate ess when `n_runs=10`
        if isinstance(runs_samples, list):
            if len(runs_samples) == 10:

                print(f"Calculating ESS from samples..")
                ess = {method: None for method in methods}

                for method in methods:

                    # projects samples to the mode from all dimensions
                    samples = np.array([draws[method] @ pdf.mode
                                       for draws in runs_samples])

                    # estimates ESS using the arviz library per dimension
                    samples_az = az.convert_to_dataset(samples)
                    ess_val = az.ess(samples_az, relative=True)
                    ess[method] = ess_val.x.values

                print(f"Saving ESS file {ess_file}")
                dump(ess, ess_file, gzip=True)

            else:
                print(f"ESS values not computed, requires `n_runs=10`")
                return None

    for method in methods:
        print(f"ESS for {method}: {ess[method]:.3%}")

    if return_ess:
        return ess


def launch_samplers(savedir, d, vmax, pdf, initial, n_samples, burnin, methods, reprod_switch):

    # load samples
    try:
        runs_samples = load(f"{savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")
        print(f"Loading file {savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")

    # run samplers
    except:
        print("File not found, starting samplers..")

        # generate fixed seeds based on `n_runs`
        if reprod_switch:
            ss = np.random.SeedSequence(48385)
            seeds = ss.spawn(n_runs)

        runs_samples = []
        for i in range(n_runs):

            # tester is instantiated based on seed
            seed = seeds[i] if reprod_switch else None
            launcher = SamplerLauncher(
                pdf, initial, n_samples, burnin, seed=seed)

            # samples saved as dict
            samples = {}
            samples['kent'] = launcher.run('kent')
            for method in methods:
                with gs.take_time(method):
                    samples[method] = launcher.run(method)

            # append for every run
            runs_samples.append(samples)

        # save a copy
        dump(runs_samples, f"{savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")
        print(f"Saving file {savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")

    return runs_samples


if __name__ == '__main__':

    n_samples = int(1e5)                  # number of samples
    burnin = int(0.1 * n_samples)         # burn-in
    n_runs = 10                           # no. of runs (ESS for `n_runs=10`)
    reprod_switch = True                  # make the samplers reproducible
    save_figs = True                      # save plots

    # dimension and lambda, ind 0 or 1 for plots with
    # 10 and 50 dimensions respectively
    ind = 0
    d, vmax = [(10, 30.), (50, 300.)][ind]

    # save directory (generate results/ dir if non-existent)
    filename = f"bingham_d{d}_vmax{int(vmax)}"
    savedir = f"results/{filename}"
    os.makedirs(savedir, exist_ok=True)

    # bingham distribution as pdf which is fixed
    pdf = random_bingham(
        d=d,
        vmax=vmax,
        vmin=0.,
        eigensystem=True,
        seed=6982
    )

    # sampler methods
    initial = pdf.mode
    methods = ('sss-reject', 'sss-shrink', 'rwmh', 'hmc')
    algos = {'sss-reject': 'geoSSS (reject)',
             'sss-shrink': 'geoSSS (shrink)',
             'rwmh': 'RWMH',
             'hmc': 'HMC'}

    # launch samplers with initial state at the mode of pdf
    runs_samples = launch_samplers(
        savedir,
        d,
        vmax,
        pdf,
        initial,
        n_samples,
        burnin,
        methods,
        reprod_switch
    )

    # calculate ess if `n_runs=10`
    bingham_ess(runs_samples, pdf, methods,
                f"{savedir}/{filename}", return_ess=False)

    # Loading the first run `ind=0` to generate plots in paper
    ind = 0
    samples = runs_samples if isinstance(
        runs_samples, dict) else runs_samples[ind]

    # plot projections
    plt.close("all")
    bins = 100
    fs = 16
    vals = samples['kent'] @ pdf.mode
    ref = list(np.histogram(vals, bins=bins, density=True))
    ref[1] = 0.5 * (ref[1][1:] + ref[1][:-1])
    plt.rc("font", size=fs)
    fig, axes = plt.subplots(1, len(methods), figsize=(len(methods) * 3, 3),
                             sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ax.set_title(algos[method], fontsize=fs)
        bins = ax.hist(samples[method] @ pdf.mode, bins=bins, density=True, alpha=0.3,
                       color='k', histtype='stepfilled')[1]
        ax.plot(*ref[::-1], color='r', lw=1, ls='--')
        ax.set_xlabel(r'$u_{d}^Tx_n$', fontsize=fs)
    fig.tight_layout()
    if save_figs:
        fig.savefig(f'{savedir}/bingham_d{d}_vmax{int(vmax)}_hist.pdf',
                    bbox_inches='tight', transparent=True)

    # trace plots
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ax.set_title(algos[method], fontsize=fs)
        ax.plot(samples[method] @ pdf.mode, alpha=0.5, color='k', lw=1)
        ax.set_xlabel(r'MCMC step $n$', fontsize=fs)
    axes[0].set_ylabel(r'$u_{d}^Tx_n$', fontsize=fs)
    fig.tight_layout()
    if save_figs:
        fig.savefig(f'{savedir}/bingham_d{d}_vmax{int(vmax)}_trace.pdf',
                    bbox_inches='tight', transparent=True)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ax.set_title(algos[method], fontsize=fs)
        ac = gs.acf(samples[method] @ pdf.mode, 1000)
        ax.plot(ac, alpha=0.7, color='k', lw=3)
        ax.axhline(0., ls='--', color='r', alpha=0.7)
        ax.set_xlabel(r'Lag', fontsize=fs)
    axes[0].set_ylabel('ACF', fontsize=fs)
    fig.tight_layout()

    if save_figs:
        fig.savefig(f'{savedir}/bingham_d{d}_vmax{int(vmax)}_acf.pdf',
                    bbox_inches='tight', transparent=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for method in methods:
        ac = gs.acf(samples[method] @ pdf.mode, 3000)
        ax.plot(ac, alpha=0.7, lw=3, label=algos[method])
    ax.legend(fontsize=fs)
    ax.axhline(0., ls='--', color='k', alpha=0.7)
    ax.set_xlabel(r'Lag', fontsize=fs)
    ax.set_ylabel('ACF', fontsize=fs)
    # hopping frequency as bar plot
    freqs = [hopping_frequency(samples[method], pdf) for method in methods]
    ax = axes[1]
    ax.set_ylabel("Hopping frequency")
    ax.bar(list(map(algos.get, methods)), freqs, color='k', alpha=0.3)
    # ax.set_ylim(None, 1.0)
    ax.semilogy()
    plt.xticks(rotation=30)
    fig.tight_layout()

    if save_figs:
        fig.savefig(f'{savedir}/bingham_d{d}_vmax{int(vmax)}_acf_v2.pdf',
                    bbox_inches='tight', transparent=True)

    # geodesic distance
    fig, axes = plt.subplots(1, len(methods), figsize=(len(methods) * 3, 3),
                             sharex=True, sharey=True)
    bins = 100
    for ax, method in zip(axes, methods):
        ax.set_title(algos[method], fontsize=fs)
        # distance between successive samples
        x = samples[method]
        dist = gs.distance(x[:-1], x[1:])
        print('average great circle distance of successive samples: '
              f'{np.mean(dist):.2f} ({method})')
        bins = ax.hist(dist, bins=bins, density=True, alpha=0.3,
                       color='k', histtype='stepfilled')[1]
        ax.set_xlabel(r'$\delta(x_{n+1}, x_n)$', fontsize=fs)
        ax.set_xticks(np.linspace(0., np.pi, 3))
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    fig.tight_layout()

    if save_figs:
        fig.savefig(f'{savedir}/bingham_d{d}_vmax{int(vmax)}_dist.pdf',
                    bbox_inches='tight', transparent=True)
