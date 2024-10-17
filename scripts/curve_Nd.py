import argparse
import logging
import os

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from csb.io import dump, load

import geosss as gs
from geosss.distributions import CurvedVonMisesFisher, Distribution
from geosss.spherical_curve import SlerpCurve, SphericalCurve

mpl.rcParams["mathtext.fontset"] = "cm"  # Use Computer Modern font

METHODS = ("sss-reject", "sss-shrink", "rwmh", "hmc")
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
}


def setup_logging(savedir: str, kappa: float, filemode: str = "a"):
    """Setting up logging

    Parameters
    ----------
    savedir : str
        log file directory
    kappa : float
        concentration parameter
    filemode : str
        'w' to overwrite the log file, 'a' to append
    """
    logpath = f"{savedir}/curve_kappa{int(kappa)}.log"
    logging.basicConfig(
        filename=logpath,
        filemode=filemode,  # 'w' to overwrite the log file, 'a' to append
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


def scatter_matrix(
    n_dim: int,
    samples: dict,
    methods: tuple,
    algos: dict,
    path: str,
    filename: str,
    savefig: bool = False,
):
    """
    Plotting scatter matrix with the corner library and adjusted label sizes
    """
    # Define font sizes
    label_size = 18  # Size for axis labels
    tick_size = 12  # Size for tick labels
    legend_size = 24  # Size for legend

    # create dir to save scatter matrices
    labels = [rf"$\mathbb{{S}}_{{{i}}}$" for i in range(n_dim)]

    # Set default font sizes for matplotlib
    plt.rcParams.update(
        {
            "font.size": tick_size,
            "axes.labelsize": label_size,
            "axes.titlesize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
        }
    )

    # Create custom labels for each dataset
    colors = ["tab:blue", "tab:orange", "tab:green", "indianred"]
    figure = plt.figure(figsize=(18, 18))

    for method, color in zip(methods, colors):

        # First corner plot for contours and 1D histograms using all samples
        figure = corner.corner(
            samples[method],
            bins=150,
            color=color,
            labels=labels,
            fig=figure,
            plot_density=False,
            plot_contours=True,  # shows the 2D histograms with contours
            contour_kwargs={"alpha": 0.8},
            plot_datapoints=False,
            levels=[0.68, 0.95],
            labelsize=label_size,
            label_kwargs={"fontsize": label_size, "labelpad": 10},
            tick_labels_size=tick_size,
            hist_kwargs={"alpha": 1.0},  # 1D histogram
            smooth1d=1.0,  # smoothens the 1D histogram
        )

        # Second corner plot for showing fewer scatter points
        figure = corner.corner(
            samples[method][::30],
            bins=50,
            color=color,
            plot_density=False,
            plot_contours=False,
            fig=figure,
            plot_datapoints=True,  # only shows the scatter points
            data_kwargs={"alpha": 0.1},
            labels=labels,
            labelsize=label_size,
            label_kwargs={"fontsize": label_size, "labelpad": 10},
            tick_labels_size=tick_size,
            hist_kwargs={"alpha": 0.0},  # 1D histogram disabled
        )

    # Create custom legend with the figure instance
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    figure.legend(
        legend_handles,
        [algos[method] for method in methods],
        loc="upper right",
        fontsize=legend_size,
    )

    # Adjust tick label sizes for all axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))
    for ax in axes.flat:
        if ax is not None:
            ax.tick_params(labelsize=tick_size)

    # save corner plot
    if savefig:
        savedir = f"{path}/corner_plots"
        os.makedirs(savedir, exist_ok=True)
        logging.info(f"Saving corner plot to {savedir}/{filename}.pdf")
        figure.savefig(f"{savedir}/{filename}.pdf", bbox_inches="tight", dpi=150)


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

    parser.add_argument(
        "--dimension",
        type=int,
        default=10,
        help="Dimension of the curve (default: 10)",
    )

    parser.add_argument(
        "-n_runs",
        "--n_runs",
        required=False,
        default=1,
        help="no. of runs per sampler",
        type=int,
    )

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Parse arguments
    args = argparser()

    # parameters
    kappa = args.kappa  # concentration parameter (default: 300.0)
    n_samples = int(args.n_samples)  # number of samples per sampler (default: 1000)
    burnin = int(0.1 * n_samples)  # burn-in
    n_dim = args.dimension  # dimensionality (default: 10)
    n_runs = args.n_runs  # sampler runs (default: 1), for ess computations `n_runs=10`

    # optional controls
    brownian_curve = True  # brownian curve or curve with fixed knots
    reprod_switch = True  # seeds samplers for reproducibility
    show_plots = True  # show the plots
    savefig = True  # save the plots
    rerun_if_samples_exists = True  # rerun even if samples file exists

    # directory to save results
    savedir = f"results/vMF_curve_{n_dim}d_kappa{int(kappa)}_brownian_curve"
    os.makedirs(savedir, exist_ok=True)
    setup_logging(savedir, kappa)

    # Define curve on the sphere
    if not brownian_curve:
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

        # Pad to match dimensionality if needed
        if n_dim > knots.shape[1]:
            knots = np.pad(knots, ((0, 0), (n_dim - knots.shape[1], 0)))

        curve = SlerpCurve(knots)

    else:
        # generates a smooth curve on the sphere with brownian motion
        knots = gs.sphere.brownian_curve_on_sphere(
            n_points=10,
            dimension=n_dim,
            step_size=0.5,  # larger step size will result in more spread out points
            seed=4562,
        )
        curve = SlerpCurve(knots)
        # curve = SlerpCurve.random_curve(n_knots=100, seed=4562, dimension=n_dim)

    # Initialize based on dimensionality
    initial = (
        np.array([0.65656515, -0.63315859, -0.40991755])
        if n_dim == 3
        else gs.sample_sphere(n_dim - 1, seed=1345)
    )

    pdf = CurvedVonMisesFisher(curve, kappa)

    # initial state fixed and samplers seeded for reproducibility
    seed_samplers = 6756 if reprod_switch else None

    # `tester` instances samplers
    launcher = gs.SamplerLauncher(pdf, initial, n_samples, burnin, seed_samplers)

    # load samples by running or loading from memory
    samples, logprob = launch_samplers(
        savedir,
        kappa,
        pdf,
        launcher,
        METHODS,
        rerun_if_samples_exists,
    )

    # plot samples on a 3d sphere
    if n_dim == 3:
        fig = visualize_samples(samples, METHODS, ALGOS, curve)
        if savefig:
            fig.savefig(
                f"{savedir}/curve_samples_kappa{int(kappa)}.pdf",
                bbox_inches="tight",
                transparent=True,
            )

    # generate figures

    # corner plot (scatter matrix)
    scatter_matrix(
        n_dim,
        samples,
        METHODS,
        ALGOS,
        savedir,
        f"curve_corner_{n_dim}d_kappa{int(kappa)}",
        savefig=True,
    )

    if show_plots:
        # autocorrelation between samples for the first three dimensions
        fs = 16
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for dim, ax in enumerate(axes):
            ax.set_title(rf"$x_{dim+1}$", fontsize=20)
            for method in METHODS:
                ac = gs.acf(samples[method][:, dim], 250)
                ax.plot(ac, alpha=0.7, lw=3, label=ALGOS[method])
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

        # geodesic distance between successive samples
        fig, axes = plt.subplots(
            1, len(METHODS), figsize=(len(METHODS) * 3, 3), sharex=True, sharey=True
        )
        bins = 100
        for ax, method in zip(axes, METHODS):
            ax.set_title(ALGOS[method], fontsize=fs)
            # distance between successive samples
            x = samples[method]
            geo_dist = gs.distance(x[:-1], x[1:])
            logging.info(
                "average great circle distance of successive samples: "
                f"{np.mean(geo_dist):.2f} ({method})"
            )
            bins = ax.hist(
                geo_dist,
                bins=bins,
                density=True,
                alpha=0.3,
                color="k",
                histtype="stepfilled",
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

        misc_plots = False
        if misc_plots:
            # autocorrelation between samples
            # NOTE: This is repeated from above with just the lag=1000
            fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
            lag = 1000
            for ax, method in zip(axes[0], METHODS):
                ax.set_title(ALGOS[method], fontsize=fs)
            for dim in range(3):
                for ax, method in zip(axes[dim], METHODS):
                    ac = gs.acf(samples[method][:, dim], lag)
                    ax.plot(ac, alpha=0.7, color="k", lw=3)
                    ax.axhline(0.0, ls="--", color="r", alpha=0.7)
            for ax in axes[-1]:
                ax.set_xlabel(r"Lag", fontsize=fs)
            for dim, ax in enumerate(axes[:, 0], 1):
                ax.set_ylabel(rf"ACF $x_{dim}$", fontsize=fs)
            ax.set_xlim(-5, 250)
            fig.suptitle("Autocorrelation between samples", fontsize=fs)
            fig.tight_layout()

            # autocorrelation between logprob of the samples
            fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
            for ax, method in zip(axes, METHODS):
                ac = gs.acf(logprob[method], 1000)
                ax.plot(ac, color="k", alpha=1.0, lw=3)
                ax.set_xlim(-5, 200)
            fig.suptitle("Autocorrelation between logprob of the samples", fontsize=fs)
            fig.tight_layout()

            bins = 50
            plt.rc("font", size=fs)
            fig, rows = plt.subplots(
                n_dim, len(METHODS), figsize=(10, 15), sharex=True, sharey=True
            )
            for dim, axes in enumerate(rows):
                vals = x[:, dim]
                # ref = list(np.histogram(vals, weights=p, bins=bins, density=True))
                # ref[1] = 0.5 * (ref[1][1:] + ref[1][:-1])
                for ax, method in zip(axes, METHODS):
                    bins = ax.hist(
                        samples[method][:, dim],
                        bins=bins,
                        density=True,
                        alpha=0.3,
                        color="k",
                        histtype="stepfilled",
                    )[1]
                    # ax.plot(*ref[::-1], color="r", lw=1, ls="--")
                    ax.set_xlabel(rf"$e_{dim}^Tx_n$", fontsize=fs)
            for ax, method in zip(rows[0], METHODS):
                ax.set_title(ALGOS[method], fontsize=fs)
            fig.tight_layout()
            if savefig:
                fig.savefig(
                    f"{savedir}/curve_hist_kappa{int(kappa)}.pdf",
                    bbox_inches="tight",
                    transparent=True,
                )
