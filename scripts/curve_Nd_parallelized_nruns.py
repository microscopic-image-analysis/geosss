import argparse
import logging
import os

import numpy as np
from csb.io import dump, load
from joblib import Parallel, delayed

import geosss as gs
from geosss.distributions import CurvedVonMisesFisher, Distribution
from geosss.spherical_curve import SlerpCurve, brownian_curve


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


def aggregate_results(n_runs, savepath_samples_base, savepath_logprob_base):
    all_samples = {}
    all_logprob = {}

    for i in range(n_runs):
        samples_i = load(f"{savepath_samples_base}_run{i}.pkl")
        logprob_i = load(f"{savepath_logprob_base}_run{i}.pkl")

        # store in a dictionary
        all_samples[f"run_{i}"] = samples_i
        all_logprob[f"run_{i}"] = logprob_i

    # Now, `all_samples` and `all_logprob` are lists containing results from all runs
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
    aggregate_results: bool = False,
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
        seed_initial_state = seeds[i]  # Alternatively, use different seeds
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

            if aggregate_results:
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
        print(f"Starting parallel sampling for {len(run_args)} runs...")
        Parallel(n_jobs=-1)(delayed(_sampler_single_run)(*args) for args in run_args)
        print("Parallel sampling completed.")

        # Load the samples from executed runs
        if aggregate_results:
            for i in runs_to_execute:

                # load the samples
                savepath_samples_i = f"{savepath_samples_base}_run{i}.pkl"
                savepath_logprob_i = f"{savepath_logprob_base}_run{i}.pkl"

                samples_i = load(savepath_samples_i)
                logprob_i = load(savepath_logprob_i)

                # store in a dictionary
                all_samples[f"run_{i}"] = samples_i
                all_logprob[f"run_{i}"] = logprob_i

    # `all_samples` and `all_logprob` contain the results from all runs
    if aggregate_results:
        return all_samples, all_logprob


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
        default=10,
        help="Number of runs per sampler",
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
    n_runs = int(args["n_runs"])  # default: 10
    burnin = 0.2  # burn-in

    # directory to save results
    if args["out_dir"] is not None:
        savedir = args["out_dir"]
    else:
        savedir = f"results/curve_{n_dim}d_kappa_{float(kappa)}"

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

    methods = ("sss-reject", "sss-shrink", "rwmh", "hmc")
    algos = {
        "sss-reject": "geoSSS (reject)",
        "sss-shrink": "geoSSS (shrink)",
        "rwmh": "RWMH",
        "hmc": "HMC",
    }
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
        aggregate_results=False,
    )


if __name__ == "__main__":

    main()
