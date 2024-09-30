# %%
import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from csb.io import dump, load

import geosss as gs
from geosss.curve import SlerpCurve, SphericalCurve
from geosss.distributions import CurvedVonMisesFisher, Distribution


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


def setup_logging(savedir: str, kappa: float):
    """Setting up logging

    Parameters
    ----------
    savedir : str
        log file directory
    kappa : float
        concentration parameter
    """
    logpath = f"{savedir}/curve_kappa{int(kappa)}_log.txt"
    logging.basicConfig(
        filename=logpath,
        filemode="w",  # 'w' to overwrite the log file, 'a' to append
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def _start_sampling(
    methods: str,
    tester: gs.SamplerLauncher,
    pdf: Distribution,
    savepath_samples: str,
    savepath_logprob: str,
):
    """just a util function to pass the output of this in a log file."""
    samples = {}
    for method in methods:
        with gs.take_time(method):
            samples[method] = tester.run(method)

            logging.info(
                "\n---------------------------------------------------------------------\n"
                f"Starting the sampler {method}\n"
                "---------------------------------------------------------------------\n"
            )

            logging.info(f"Gradient calls for {method}: {pdf.gradient.num_calls}")
            logging.info(f"Logprob calls for {method}: {pdf.log_prob.num_calls}")

            if method == "sss-reject":
                logging.info(f"Rejected samples for {method}: {tester.rsss.n_reject}")

            if method == "sss-shrink":
                logging.info(f"Rejected samples for {method}: {tester.ssss.n_reject}")

    logging.info(
        "\n---------------------------------------------------------------------\n"
        "---------------------------------------------------------------------\n"
    )

    logprob = {}
    for method in methods:
        logprob[method] = pdf.log_prob(samples[method])

    dump(samples, savepath_samples)
    dump(logprob, savepath_logprob)

    return samples, logprob


def launch_samplers(
    savedir: str,
    kappa: float,
    pdf: Distribution,
    tester: gs.SamplerLauncher,
    methods: dict,
    rerun_if_file_exists: bool = False,
):
    """just an interface to load or run samplers"""

    savepath_samples = f"{savedir}/curve_samples_kappa{int(kappa)}.pkl"
    savepath_logprob = f"{savedir}/curve_logprob_kappa{int(kappa)}.pkl"

    if (
        not rerun_if_file_exists
        and os.path.exists(savepath_samples)
        and os.path.exists(savepath_logprob)
    ):
        samples = load(savepath_samples)
        logging.info(f"Loading file {savepath_samples}")

        logprob = load(savepath_logprob)
        logging.info(f"Loading file {savepath_logprob}")
    else:
        samples, logprob = _start_sampling(
            methods,
            tester,
            pdf,
            savepath_samples,
            savepath_logprob,
        )

    return samples, logprob


def visualize_samples(
    samples: dict,
    methods: tuple,
    algos: dict,
    curve: SphericalCurve,
):
    """visualize samples on a 3d sphere"""
    phi, theta = np.mgrid[0 : np.pi : 20j, 0 : 2 * np.pi : 30j]
    euler = (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))
    t = np.linspace(0, 1, 1_000)  # points on curve

    fig, axes = plt.subplots(
        2, 2, figsize=(8, 8), subplot_kw=dict(projection="3d"), sharex=True, sharey=True
    )

    for ax, method in zip(axes.flat, methods):
        ax.computed_zorder = False
        ax.plot_wireframe(*euler, color="green", alpha=0.06, zorder=1)
        ax.plot_surface(*euler, cmap="viridis", alpha=0.07, zorder=2)
        x = samples[method][: int(1e4)]
        ax.set_title(algos[method])
        ax.plot(*curve(t).T, color="r", alpha=0.9, lw=3, zorder=3)
        ax.scatter(*x.T, c="k", s=1, alpha=0.08, zorder=4)
        ax.set_aspect("equal")
        ax.view_init(-140, 20)
    fig.tight_layout()

    return fig


def argparser():
    parser = argparse.ArgumentParser(
        description="Process parameters for the curve generation."
    )

    # Add arguments for kappa and n_samples
    parser.add_argument(
        "--kappa",
        type=float,
        default=300.0,
        help="Concentration parameter (default: 300.0)",
    )
    parser.add_argument(
        "--n_samples",
        type=float,
        default=1e3,
        help="Number of samples per sampler (default: 1000)",
    )

    # Parse arguments
    args = parser.parse_args()

    return args


# %%
if __name__ == "__main__":

    # Parse arguments
    args = argparser()

    # parameters
    kappa = args.kappa  # concentration parameter (default: 300.0)
    n_samples = int(args.n_samples)  # number of samples per sampler (default: 1000)
    burnin = int(0.1 * n_samples)  # burn-in
    d = 10  # dimensionality

    # optional controls
    fix_curve = True  # fix curve (target)
    reprod_switch = True  # seeds samplers for reproducibility
    savefig = False  # save the plots
    rerun_if_samples_exists = True  # rerun even if samples file exists

    # directory to save results
    savedir = f"results_temp/vMF_curve_{d}d_kappa{int(kappa)}"
    os.makedirs(savedir, exist_ok=True)
    setup_logging(savedir, kappa)

    # define curve on the sphere
    if fix_curve:
        # NOTE: keeping these knots to match with previous paper version
        knots = np.array(
            [
                [-0.25882694, 0.95006168, 0.17433133],
                [0.14557335, 0.61236727, 0.77705516],
                [-0.7973001, -0.25170369, 0.54859622],
                [0.03172733, -0.71944851, 0.69382074],
                [0.56217797, -0.29453368, 0.77279094],
                [0.80883044, 0.1316755, 0.57310983],
                [0.98981463, 0.03039439, -0.13907979],
                [0.81592815, 0.04723609, -0.57622045],
                [0.36888235, 0.400026, -0.83899047],
                [-0.6770828, 0.05213374, -0.73405787],
            ]
        )
        # padding the original 2-sphere knots to N-sphere
        d_orig = knots.shape[1]

        # initial fixed state
        initial = np.array([0.65656515, -0.63315859, -0.40991755])
        if d > d_orig:

            # lifts the sphere to higher dimensions by padding with zeros
            delta_d = d - d_orig
            knots = np.pad(knots, ((0, 0), (delta_d, 0)))

            # initial fixed state
            seed_init_state = 1345
            initial = gs.sample_sphere(d - 1, seed=seed_init_state)
        curve = SlerpCurve(knots)

    else:
        curve = SlerpCurve.random_curve(n_knots=10, seed=None, dimension=d)

    pdf = CurvedVonMisesFisher(curve, kappa)

    # initial state fixed and samplers seeded for reproducibility
    seed_samplers = 6756 if reprod_switch else None

    # `tester` instances samplers
    launcher = gs.SamplerLauncher(pdf, initial, n_samples, burnin, seed_samplers)
    methods = ("sss-reject", "sss-shrink", "rwmh", "hmc")
    algos = {
        "sss-reject": "geoSSS (reject)",
        "sss-shrink": "geoSSS (shrink)",
        "rwmh": "RWMH",
        "hmc": "HMC",
    }

    # %%
    # load samples by running or loading from memory
    samples, logprob = launch_samplers(
        savedir,
        kappa,
        pdf,
        launcher,
        methods,
        rerun_if_samples_exists,
    )

    # plot samples on a 3d sphere
    if d == 3:
        fig = visualize_samples(samples, methods, algos, curve)
        if savefig:
            fig.savefig(
                f"{savedir}/curve_samples_kappa{int(kappa)}.pdf",
                bbox_inches="tight",
                transparent=True,
            )

    # generate figures
    fs = 16
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for d, ax in enumerate(axes):
        ax.set_title(rf"$x_{d+1}$", fontsize=20)
        for method in methods:
            ac = gs.acf(samples[method][:, d], 250)
            ax.plot(ac, alpha=0.7, lw=3, label=algos[method])
        ax.axhline(0.0, ls="--", color="k", alpha=0.7)
        ax.set_xlabel(r"lag", fontsize=fs)
    axes[0].set_ylabel("ACF", fontsize=fs)
    ax.legend(fontsize=fs)
    fig.tight_layout()
    if savefig:
        fig.savefig(
            f"{savedir}/curve_acf_kappa{int(kappa)}.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    # geodesic distance
    fig, axes = plt.subplots(
        1, len(methods), figsize=(len(methods) * 3, 3), sharex=True, sharey=True
    )
    bins = 100
    for ax, method in zip(axes, methods):
        ax.set_title(algos[method], fontsize=fs)
        # distance between successive samples
        x = samples[method]
        d = gs.distance(x[:-1], x[1:])
        logging.info(
            "average great circle distance of successive samples: "
            f"{np.mean(d):.2f} ({method})"
        )
        bins = ax.hist(
            d, bins=bins, density=True, alpha=0.3, color="k", histtype="stepfilled"
        )[1]
        ax.set_xlabel(r"$\delta(x_{n+1}, x_n)$", fontsize=fs)
        ax.set_xticks(np.linspace(0.0, np.pi, 3))
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        ax.semilogy()
    fig.tight_layout()
    if savefig:
        fig.savefig(
            f"{savedir}/curve_dist_kappa{int(kappa)}.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    # autocorrelation between samples
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
    for ax, method in zip(axes[0], methods):
        ax.set_title(algos[method], fontsize=fs)
    for d in range(3):
        for ax, method in zip(axes[d], methods):
            ac = gs.acf(samples[method][:, d], 1000)
            ax.plot(ac, alpha=0.7, color="k", lw=3)
            ax.axhline(0.0, ls="--", color="r", alpha=0.7)
    for ax in axes[-1]:
        ax.set_xlabel(r"Lag", fontsize=fs)
    for d, ax in enumerate(axes[:, 0], 1):
        ax.set_ylabel(rf"ACF $x_{d}$", fontsize=fs)
    ax.set_xlim(-5, 250)
    fig.suptitle("Autocorrelation between samples", fontsize=fs)
    fig.tight_layout()

    # autocorrelation between logprob of the samples
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ac = gs.acf(logprob[method], 1000)
        ax.plot(ac, color="k", alpha=1.0, lw=3)
        ax.set_xlim(-5, 200)
    fig.suptitle("Autocorrelation between logprob of the samples", fontsize=fs)
    fig.tight_layout()

    bins = 50
    plt.rc("font", size=fs)
    fig, rows = plt.subplots(
        3, len(methods), figsize=(12, 10), sharex=True, sharey=True
    )
    for i, axes in enumerate(rows):
        vals = x[:, i]
        # ref = list(np.histogram(vals, weights=p, bins=bins, density=True))
        # ref[1] = 0.5 * (ref[1][1:] + ref[1][:-1])
        for ax, method in zip(axes, methods):
            bins = ax.hist(
                samples[method][burnin:, i],
                bins=bins,
                density=True,
                alpha=0.3,
                color="k",
                histtype="stepfilled",
            )[1]
            # ax.plot(*ref[::-1], color="r", lw=1, ls="--")
            ax.set_xlabel(rf"$e_{i}^Tx_n$", fontsize=fs)
    for ax, method in zip(rows[0], methods):
        ax.set_title(algos[method], fontsize=fs)
    fig.tight_layout()
    if savefig:
        fig.savefig(
            f"{savedir}/curve_hist_kappa{int(kappa)}.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    plt.show()
