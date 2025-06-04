import argparse
import logging
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from csb.io import dump, load

import geosss as gs
from geosss.distributions import CurvedVonMisesFisher, Distribution
from geosss.spherical_curve import SlerpCurve, brownian_curve

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
    samples_filename: str = None,
    samples_logprob_filename: str = None,
):
    """just an interface to load or run samplers"""

    if samples_filename is None:
        samples_filename = f"curve_samples_kappa{int(kappa)}.pkl"
    if samples_logprob_filename is None:
        samples_logprob_filename = f"curve_logprob_kappa{int(kappa)}.pkl"

    savepath_samples = f"{savedir}/{samples_filename}"
    savepath_logprob = f"{savedir}/{samples_logprob_filename}"

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


def acf_geodist_plot(
    samples,
    methods,
    algos,
    savedir,
    filename="curve_acf_distplot",
    savefig=True,
):
    # Suppress FutureWarnings (optional)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Define your methods and corresponding colors
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    method_color_dict = dict(zip(methods, colors))

    # Font size for labels and titles
    fs = 16

    # Create the figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ### First Subplot: Autocorrelation Function ###
    ax1 = axes[0]

    for method, color in zip(methods, colors):
        ac = gs.acf(samples[method][:, 0], 4000)
        ax1.plot(ac, alpha=0.7, lw=3, label=algos[method], color=color)

    ax1.axhline(0.0, ls="--", color="k", alpha=0.7)
    ax1.set_xlabel(r"Lag", fontsize=fs)
    ax1.set_ylabel("ACF", fontsize=fs)
    ax1.tick_params(axis="both", which="major", labelsize=fs)
    ax1.legend(fontsize=fs, loc="upper right")

    ### Second Subplot: Geodesic Distance Histogram ###
    ax2 = axes[1]

    # Prepare the data for the histogram
    geo_dist_list = []
    for method in methods:
        x = samples[method]
        # Compute geodesic distances between successive samples
        geo_dist = gs.sphere.distance(x[:-1], x[1:])
        # Check for Inf or NaN values
        if not np.all(np.isfinite(geo_dist)):
            logging.warning(
                f"Infinite or NaN values found in geo_dist for method {method}"
            )
            # Remove or handle these values
            geo_dist = geo_dist[np.isfinite(geo_dist)]
        logging.info(
            "average great circle distance of successive samples: "
            f"{np.mean(geo_dist):.2f} ({method})"
        )
        # Create a DataFrame for the current method
        df_method = pd.DataFrame({"geo_dist": geo_dist, "method": method})
        geo_dist_list.append(df_method)

    # Combine all DataFrames into one
    df_geo_dist = pd.concat(geo_dist_list, ignore_index=True)

    # Set the style
    sns.set_style("white")  # Remove the background grid

    # Create the histogram plot using Seaborn
    sns.histplot(
        data=df_geo_dist,
        x="geo_dist",
        hue="method",
        bins=400,
        stat="density",
        element="step",  # Use 'bars' for filled histograms
        fill=True,  # Set to True for filled histograms
        common_norm=False,  # Normalize each histogram independently
        linewidth=1.5,  # Adjust line width for better visibility
        alpha=0.4,
        ax=ax2,
        palette=method_color_dict,
        legend=True,  # Ensure legend is enabled
    )

    # Customize the x-axis labels and ticks
    ax2.set_xlabel(r"$\delta(x_{n+1}, x_n)$", fontsize=20)
    ax2.set_xticks([0, np.pi / 2, np.pi])
    ax2.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"], fontsize=20)
    ax2.tick_params(axis="both", which="major", labelsize=fs)

    # Set y-scale to logarithmic
    ax2.set_yscale("log")
    ax2.set_ylabel(None)  # Remove the y-axis label
    ax2.set_xlim(0, np.pi)

    # Customize the legend
    leg = ax2.get_legend()
    if leg is not None:
        leg.set_title(None)  # Remove the legend title
        for t in leg.texts:
            t.set_fontsize(fs)
        # Optionally, adjust the legend location
        leg.set_bbox_to_anchor((1, 1))
    else:
        logging.warning("Legend not found in ax2.")

    # Adjust layout
    fig.tight_layout()

    if savefig:
        logging.info(
            f"Saving ACF and geodesic distance plot to {savedir}/{filename}.pdf"
        )
        savedir_acf_dist = f"{savedir}/dist_acf_plots"
        os.makedirs(savedir_acf_dist, exist_ok=True)
        fig.savefig(
            f"{savedir_acf_dist}/{filename}.pdf",
            bbox_inches="tight",
            transparent=True,
            dpi=150,
        )


def argparser():
    parser = argparse.ArgumentParser(
        description="Process parameters for the curve generation."
    )

    # Add arguments for kappa and n_samples
    parser.add_argument(
        "--kappa",
        type=float,
        default=800.0,
        help="Concentration parameter (default: 300.0)",
    )
    parser.add_argument(
        "--n_samples",
        type=float,
        default=1e6,
        help="Number of samples per sampler (default: 1000)",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=5,
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
    is_brownian_curve = True  # brownian curve or curve with fixed knots
    reprod_switch = True  # seeds samplers for reproducibility
    show_plots = False  # show the plots
    savefig = True  # save the plots
    rerun_if_samples_exists = False  # rerun even if samples file exists

    # directory to save results
    savedir = f"results/curve_{n_dim}d_vary_kappa_nruns_10/curve_{n_dim}d_kappa_{float(kappa)}"
    os.makedirs(savedir, exist_ok=True)
    setup_logging(savedir, kappa)

    # Define curve on the sphere
    if not is_brownian_curve:
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

    else:
        # generates a smooth curve on the sphere with brownian motion
        knots = brownian_curve(
            n_points=10,
            dimension=n_dim,
            step_size=0.5,  # larger step size will result in more spread out points
            seed=4562,
        )

    curve = SlerpCurve(knots)

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

    # filenames for samples and logprob
    samples_filename = f"curve_samples_{n_dim}d_kappa_{float(kappa)}_run0.pkl"
    samples_logprob_filename = f"curve_logprob_{n_dim}d_kappa_{float(kappa)}_run0.pkl"

    # load samples by running or loading from memory
    samples, logprob = launch_samplers(
        savedir,
        kappa,
        pdf,
        launcher,
        METHODS,
        rerun_if_samples_exists,
        samples_filename,
        samples_logprob_filename,
    )

    # generate figures
    acf_geodist_plot(
        samples,
        METHODS,
        ALGOS,
        savedir,
        f"curve_acf_geodist_{n_dim}d_kappa{int(kappa)}",
        savefig=True,
    )
