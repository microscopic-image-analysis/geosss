import argparse
import logging
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

import geosss as gs
from geosss.distributions import CurvedVonMisesFisher, Distribution
from geosss.io import dump, load
from geosss.spherical_curve import SlerpCurve, brownian_curve

mpl.rcParams["mathtext.fontset"] = "cm"  # Use Computer Modern font

METHODS = ("sss-reject", "sss-shrink", "rwmh", "hmc")
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
}


def setup_logging(
    savedir: str = None, kappa: float = None, filemode: str = "a", logpath: str = None
):
    """Setting up logging

    Parameters
    ----------
    savedir : str
        log file directory
    kappa : float
        concentration parameter
    filemode : str
        'w' to overwrite the log file, 'a' to append
    logpath : str
        full path to log file (overrides savedir/kappa combination)
    """
    if logpath is None:
        if savedir is None or kappa is None:
            raise ValueError(
                "Either logpath or both savedir and kappa must be provided"
            )
        logpath = f"{savedir}/curve_kappa{int(kappa)}.log"

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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


def _sampler_single_run(
    methods: list,
    pdf: Distribution,
    n_samples: int,
    burnin: int,
    savepath_samples: str,
    savepath_logprob: str,
    seed_sampler: int = None,
    seed_initial_state: int = None,
    logpath: str = None,
):
    """Runs a single sampler and saves the results."""

    # Set up logging for this process
    setup_logging(logpath=logpath)

    # Generate initial state
    init_state = gs.sample_sphere(pdf.d - 1, seed=seed_initial_state)
    logging.info(f"Initial state for this run: {init_state}")

    # Create a new SamplerLauncher instance
    launcher = gs.SamplerLauncher(pdf, init_state, n_samples, burnin, seed_sampler)

    samples = {}
    logprob = {}

    for method in methods:
        with gs.take_time(method):
            samples[method] = launcher.run(method)

            logging.info(
                "\n---------------------------------------------------------------------\n"
                f"Starting the sampler {method}\n"
                "---------------------------------------------------------------------\n"
            )

            logging.info(f"Gradient calls for {method}: {pdf.gradient.num_calls}")
            logging.info(f"Logprob calls for {method}: {pdf.log_prob.num_calls}")

            if method == "sss-reject":
                logging.info(f"Rejected samples for {method}: {launcher.rsss.n_reject}")

            if method == "sss-shrink":
                logging.info(f"Rejected samples for {method}: {launcher.ssss.n_reject}")

    logging.info(
        "\n---------------------------------------------------------------------\n"
        "---------------------------------------------------------------------\n"
    )

    for method in methods:
        logprob[method] = pdf.log_prob(samples[method])

    # Save samples and log probabilities
    dump(samples, savepath_samples)
    dump(logprob, savepath_logprob)


def _start_sampling(
    methods: list,
    tester: gs.SamplerLauncher,
    pdf: Distribution,
    savepath_samples: str,
    savepath_logprob: str,
):
    """Utility function for single-run sampling (legacy compatibility)."""
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


def aggregate_results(n_runs, savepath_samples_base, savepath_logprob_base):
    """Aggregate results from multiple runs."""
    all_samples = {}
    all_logprob = {}

    for i in range(n_runs):
        samples_i = load(f"{savepath_samples_base}_run{i}.pkl")
        logprob_i = load(f"{savepath_logprob_base}_run{i}.pkl")

        # store in a dictionary
        all_samples[f"run_{i}"] = samples_i
        all_logprob[f"run_{i}"] = logprob_i

    return all_samples, all_logprob


def load_or_launch_samplers(
    methods: list,
    pdf: Distribution,
    n_samples: int,
    burnin: int,
    savepath_samples_base: str,
    savepath_logprob_base: str,
    reprod_switch: bool = True,
    n_runs: int = 10,
    seed_sequence: int = 48385,
    rerun_if_samples_exists: bool = False,
    aggregate_results_flag: bool = False,
    initial_state: np.ndarray = None,
):
    """Launches parallel sampling runs, loading existing samples if available and permitted."""

    # Extract savedir from savepath_samples_base
    savedir = os.path.dirname(savepath_samples_base)

    # Generate fixed seeds based on `n_runs` for reproducibility
    if reprod_switch:
        ss = np.random.SeedSequence(seed_sequence)
        seeds = ss.spawn(n_runs)
    else:
        seeds = [None] * n_runs

    # Prepare arguments for each run
    run_args = []

    # Keep track of which runs need to be executed
    runs_to_execute = []

    # For collecting samples and logprob
    all_samples = {}
    all_logprob = {}

    for i in range(n_runs):
        seed_sampler = seeds[i]
        seed_initial_state = seeds[i] if initial_state is None else None
        savepath_samples_i = f"{savepath_samples_base}_run{i}.pkl"
        savepath_logprob_i = f"{savepath_logprob_base}_run{i}.pkl"
        logpath_i = f"{savedir}/curve_{pdf.d}d_kappa{int(pdf.kappa)}_run{i}.log"

        if_samples_exist = os.path.exists(savepath_samples_i) and os.path.exists(
            savepath_logprob_i
        )

        if not rerun_if_samples_exists and if_samples_exist:
            # Load existing samples
            samples_i = load(savepath_samples_i)
            logprob_i = load(savepath_logprob_i)

            if aggregate_results_flag:
                all_samples[f"run_{i}"] = samples_i
                all_logprob[f"run_{i}"] = logprob_i
                logging.info(
                    f"Loading existing samples from {savepath_samples_i} and {savepath_logprob_i}"
                )
        else:
            # Need to execute this run
            run_args.append(
                (
                    methods,
                    pdf,
                    n_samples,
                    burnin,
                    savepath_samples_i,
                    savepath_logprob_i,
                    seed_sampler,
                    seed_initial_state,
                    logpath_i,
                )
            )
            runs_to_execute.append(i)

    # Execute runs in parallel if there are any to execute
    if run_args:
        if n_runs > 1:
            print(f"Starting parallel sampling for {len(run_args)} runs...")
            Parallel(n_jobs=-1)(
                delayed(_sampler_single_run)(*args) for args in run_args
            )
            print("Parallel sampling completed.")
        else:
            # Single run - execute directly without parallelization
            print("Starting single run sampling...")
            _sampler_single_run(*run_args[0])
            print("Single run sampling completed.")

        # Load the samples from executed runs
        if aggregate_results_flag:
            for i in runs_to_execute:
                # load the samples
                savepath_samples_i = f"{savepath_samples_base}_run{i}.pkl"
                savepath_logprob_i = f"{savepath_logprob_base}_run{i}.pkl"

                samples_i = load(savepath_samples_i)
                logprob_i = load(savepath_logprob_i)

                # store in a dictionary
                all_samples[f"run_{i}"] = samples_i
                all_logprob[f"run_{i}"] = logprob_i

    # Return aggregated results if requested
    if aggregate_results_flag:
        return all_samples, all_logprob


def launch_samplers(
    savedir: str,
    kappa: float,
    pdf: Distribution,
    tester: gs.SamplerLauncher,
    methods: tuple,
    rerun_if_file_exists: bool = False,
    samples_filename: str = None,
    samples_logprob_filename: str = None,
):
    """Interface to load or run samplers (legacy single-run compatibility)."""

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
    """Generate autocorrelation and geodesic distance plots."""
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
        default=500.0,
        help="Concentration parameter (default: 500.0)",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=int(1e3),
        help="Number of samples per sampler (default: 1000)",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=10,
        help="Dimension of the curve (default: 10)",
    )

    parser.add_argument(
        "--n_runs",
        required=False,
        default=1,
        help="Number of runs per sampler (default: 1 for single run, 10+ for parallel)",
        type=int,
    )

    # Add argument for output directory
    parser.add_argument(
        "--out_dir",
        required=False,
        help="Main output directory",
        default=None,
        type=str,
    )

    # Add argument for plotting
    parser.add_argument(
        "--generate_plots",
        required=False,
        action="store_true",
        help="Generate ACF and geodesic distance plots (only works for single runs)",
    )

    # Parse arguments
    args = vars(parser.parse_args())

    return args


def main():
    # Parse arguments
    args = argparser()

    # Ensure correct data types
    n_dim = int(args["dimension"])  # default: 10
    kappa = float(args["kappa"])  # default: 500
    n_samples = int(args["n_samples"])  # default: 1000
    n_runs = int(args["n_runs"])  # default: 1
    generate_plots = args["generate_plots"]  # default: False
    burnin = 0.2 if n_runs > 1 else int(0.1 * n_samples)  # burn-in

    # directory to save results
    if args["out_dir"] is not None:
        savedir = args["out_dir"]
    else:
        if n_runs > 1:
            savedir = f"results/curve_{n_dim}d_kappa_{float(kappa)}"
        else:
            savedir = f"results/curve_{n_dim}d_vary_kappa_nruns_10/curve_{n_dim}d_kappa_{float(kappa)}"

    # Create the directory if it doesn't exist
    os.makedirs(savedir, exist_ok=True)

    msg = "Computations only for 2-sphere and above"
    assert n_dim >= 3, msg

    # optional controls
    is_brownian_curve = True  # brownian curve or curve with fixed knots
    reprod_switch = True  # seeds samplers for reproducibility
    rerun_if_samples_exists = False  # rerun even if samples file exists

    # creating a target as a curve on the sphere
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

    logging.info(f"Target curve: {knots}")
    # defining the curve on the sphere
    curve = SlerpCurve(knots)

    # defining this curve as vMF distribution
    pdf = CurvedVonMisesFisher(curve, kappa)

    methods = METHODS

    if n_runs > 1:
        # Multi-run mode (parallelized)
        setup_logging(savedir, kappa)

        savepath_samples_base = f"{savedir}/curve_samples_{n_dim}d_kappa_{float(kappa)}"
        savepath_logprob_base = f"{savedir}/curve_logprob_{n_dim}d_kappa_{float(kappa)}"

        load_or_launch_samplers(
            methods,
            pdf,
            n_samples,
            burnin,
            savepath_samples_base,
            savepath_logprob_base,
            reprod_switch,
            n_runs,
            seed_sequence=48385,
            rerun_if_samples_exists=rerun_if_samples_exists,
            aggregate_results_flag=False,
        )

        if generate_plots:
            print(
                "Warning: Plot generation is only supported for single runs (n_runs=1)"
            )
            print("To generate plots, run with --n_runs 1 --generate_plots")

    else:
        # Single-run mode (with optional plotting)
        setup_logging(savedir, kappa)

        # Initialize based on dimensionality
        initial = (
            np.array([0.65656515, -0.63315859, -0.40991755])
            if n_dim == 3
            else gs.sample_sphere(n_dim - 1, seed=1345)
        )

        # initial state fixed and samplers seeded for reproducibility
        seed_samplers = 6756 if reprod_switch else None

        # `tester` instances samplers
        launcher = gs.SamplerLauncher(pdf, initial, n_samples, burnin, seed_samplers)

        # filenames for samples and logprob
        samples_filename = f"curve_samples_{n_dim}d_kappa_{float(kappa)}_run0.pkl"
        samples_logprob_filename = (
            f"curve_logprob_{n_dim}d_kappa_{float(kappa)}_run0.pkl"
        )

        # load samples by running or loading from memory
        samples, logprob = launch_samplers(
            savedir,
            kappa,
            pdf,
            launcher,
            methods,
            rerun_if_samples_exists,
            samples_filename,
            samples_logprob_filename,
        )

        # generate figures if requested
        if generate_plots:
            acf_geodist_plot(
                samples,
                methods,
                ALGOS,
                savedir,
                f"curve_acf_geodist_{n_dim}d_kappa{int(kappa)}",
                savefig=True,
            )


if __name__ == "__main__":
    main()
