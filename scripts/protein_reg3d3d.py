# To test the 3D-3D registration of two protein structures
import argparse
import logging
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from geosss import (
    MetropolisHastings,
    RejectionSphericalSliceSampler,
    ShrinkageSphericalSliceSampler,
    SphericalHMC,
)
from geosss.pointcloud import PointCloud
from geosss.registration import CoherentPointDrift, Registration
from geosss.sphere import sample_sphere
from geosss.sphere_tesselation import tessellate_rotations
from geosss.utils import take_time


def argparser():
    parser = argparse.ArgumentParser(description="Process parameters for the sampling.")

    parser.add_argument(
        "--n_samples",
        type=int,
        default=int(2e3),
        help="Number of samples per sampler (default: 2000)",
    )

    parser.add_argument(
        "--burnin",
        type=float,
        default=0.0,
        help="Fraction of burn-in samples per sampler (default: 0.0)",
    )

    parser.add_argument(
        "--n_chains",
        required=False,
        default=200,
        help="Number of chains for every sampler",
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

    # Add argument to specify number of parallel jobs
    parser.add_argument(
        "--n_jobs",
        required=False,
        default=-1,
        help="Number of parallel jobs to run (-1 for all available cores)",
        type=int,
    )

    # Parse arguments
    args = vars(parser.parse_args())

    return args


def setup_logging(logpath: str, filemode: str = "a"):
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


class SamplerLauncher:
    """Just an interface for launching all the samplers"""

    def __init__(
        self,
        pdf,
        initial,
        n_samples,
        burnin=0.2,
        seed=None,
        return_all_samples=False,
    ):
        self.pdf = pdf
        self.initial = initial
        self.n_samples = n_samples
        self.burnin = burnin
        self.seed = seed
        self.return_all_samples = return_all_samples

    def run_sss_reject(self):
        sampler = RejectionSphericalSliceSampler(
            self.pdf,
            self.initial,
            self.seed,
        )
        self.rsss = sampler

        return sampler.sample(
            self.n_samples,
            burnin=self.burnin,
            return_all_samples=self.return_all_samples,
        )

    def run_sss_shrink(self):
        sampler = ShrinkageSphericalSliceSampler(
            self.pdf,
            self.initial,
            self.seed,
        )
        self.ssss = sampler

        return sampler.sample(
            self.n_samples,
            burnin=self.burnin,
            return_all_samples=self.return_all_samples,
        )

    def run_rwmh(self, stepsize=1e-1):
        sampler = MetropolisHastings(
            self.pdf,
            self.initial,
            self.seed,
            stepsize=stepsize,
        )
        self.rwmh = sampler

        logging.info(f"Original RWMH stepsize: {self.rwmh.stepsize}")
        samples = sampler.sample(
            self.n_samples * 6,
            burnin=self.burnin,
            return_all_samples=self.return_all_samples,
        )
        logging.info(f"New RWMH stepsize: {self.rwmh.stepsize}")
        return samples

    def run_hmc(self, stepsize=1e-1):
        sampler = SphericalHMC(
            self.pdf,
            self.initial,
            self.seed,
            stepsize=stepsize,
            n_steps=50,
        )
        self.hmc = sampler

        logging.info(f"Original HMC stepsize: {self.hmc.stepsize}")
        samples = sampler.sample(
            self.n_samples,
            burnin=self.burnin,
            return_all_samples=self.return_all_samples,
        )
        logging.info(f"New HMC stepsize: {self.hmc.stepsize}")
        return samples

    def run(
        self,
        method,
        stepsize_rwmh=1e-1,
        stepsize_hmc=1e-3,
    ):
        if method == "sss-reject":
            return self.run_sss_reject()
        elif method == "sss-shrink":
            return self.run_sss_shrink()
        elif method == "rwmh":
            return self.run_rwmh(stepsize=stepsize_rwmh)
        elif method == "hmc":
            return self.run_hmc(stepsize=stepsize_hmc)
        else:
            raise ValueError(f"method {method} not known")


def _sampler_single_run(
    methods: list,
    pdf: Registration,
    n_samples: int,
    burnin: float,
    seed_sampler: int = None,
    seed_initial_state: int = None,
    logpath: str = None,
    return_all_samples: bool = False,
    stepsize_rwmh: float = 1e-1,
    stepsize_hmc: float = 1e-3,
):
    """Runs a single sampler and returns the results."""
    # Set up logging for this process
    setup_logging(logpath=logpath)

    # Generate initial state
    init_state = sample_sphere(d=3, seed=seed_initial_state)
    logging.info(f"Initial state for this run: {init_state}")

    # Create a new SamplerLauncher instance
    launcher = SamplerLauncher(
        pdf,
        init_state,
        n_samples,
        burnin,
        seed_sampler,
        return_all_samples,
    )

    samples = {}
    logprob = {}

    for method in methods:
        with take_time(method):
            # Sample quaternions
            samples[method] = launcher.run(method, stepsize_rwmh, stepsize_hmc)

            # Evaluate the density at the sampled quaternions
            logprob[method] = np.array([pdf.log_prob(draw) for draw in samples[method]])

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

    return samples, logprob


def launch_sampling_nchains(
    methods: list,
    pdf: Registration,
    n_samples: int,
    burnin: float,
    hdf5_filepath: str,
    reprod_switch: bool = True,
    n_chains: int = 10,
    seed_sequence: int = 48385,
    return_chains: bool = False,
    n_jobs: int = None,
    return_all_samples: bool = False,
    stepsize_rwmh: float = 1e-1,
    stepsize_hmc: float = 1e-3,
):
    """Original function that parallelizes only over chains, kept for backward compatibility."""

    # Generate fixed seeds based on `n_chains` for reproducibility
    if reprod_switch:
        ss = np.random.SeedSequence(seed_sequence)
        seeds = ss.spawn(n_chains)
    else:
        seeds = [None] * n_chains

    # initial state
    ss_init = np.random.SeedSequence(seed_sequence + 100)
    seeds_init = ss_init.spawn(n_chains)  # Collect the integer seeds

    # Initialize dictionaries to collect samples and logprobs
    all_samples = {method: [] for method in methods}
    all_logprob = {method: [] for method in methods}

    # Prepare arguments for each chain
    chain_args = []
    n_jobs = n_chains if n_jobs is None else n_jobs
    for i in range(n_chains):
        seed_sampler = seeds[i]
        seed_initial_state = seeds_init[i]  # Alternatively, use different seeds
        logpath_i = f"{os.path.splitext(hdf5_filepath)[0]}_chain{i}.log"

        chain_args.append(
            (
                methods,
                pdf,
                n_samples,
                burnin,
                seed_sampler,
                seed_initial_state,
                logpath_i,
                return_all_samples,
                stepsize_rwmh,
                stepsize_hmc,
            )
        )

    # Execute runs in parallel
    print(f"Starting parallel sampling for {n_chains} chains...")
    results = Parallel(n_jobs=n_chains)(
        delayed(_sampler_single_run)(*args) for args in chain_args
    )
    print("Parallel sampling completed.")

    # Collect the results
    for chain_idx, (samples_chain, logprob_chain) in enumerate(results):
        for method in methods:
            all_samples[method].append(samples_chain[method])
            all_logprob[method].append(logprob_chain[method])

    # Convert lists to arrays of shape (n_chains, n_samples, 4) and (n_chains, n_samples)
    for method in methods:
        all_samples[method] = np.stack(all_samples[method], axis=0)
        all_logprob[method] = np.stack(all_logprob[method], axis=0)

    # Save to HDF5 file
    with h5py.File(hdf5_filepath, "w") as hf:
        # Save seeds
        hf.create_dataset("seed_sequence", data=seed_sequence)

        # Save samples
        samples_group = hf.create_group("samples")
        for method in methods:
            samples_group.create_dataset(method, data=all_samples[method])

        # Save logprobs
        logprob_group = hf.create_group("logprobs")
        for method in methods:
            logprob_group.create_dataset(method, data=all_logprob[method])

    print(f"Results saved to {hdf5_filepath}")

    if return_chains:
        return all_samples, all_logprob


def plot_heatmap_logprobs(logprobs_chains, methods, savedir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, method in zip(axes.flat, methods):
        im = ax.imshow(logprobs_chains[method], aspect="auto", cmap="nipy_spectral")
        ax.set_xlabel("MCMC steps")
        ax.set_ylabel("Chains")
        ax.set_title(method, fontsize=13)
        plt.colorbar(im, ax=ax)
    fig.suptitle("Log-probabilities for every chain", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{savedir}/protein_reg3d3d_logprobs.png", dpi=150)


def plot_hist_logprobs(log_probs_C600, logprobs_chains, methods, savedir, threshold):
    # compare histogram of the log probs for all the methods
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(16, 3))
    colors = ["blue", "tab:orange", "tab:green", "tab:red"]

    axes[0].hist(
        log_probs_C600,
        bins=50,
        alpha=0.5,
        label="tesselations",
        color="k",
        density=True,
    )
    print(f"best systematic-search log prob: {np.max(log_probs_C600)}")
    axes[0].axvline(threshold, color="k", ls="--", label="optimum")
    axes[0].set_title("systematic search via \n600-cell")
    axes[0].ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
    axes[0].set_yscale("log")
    for i, (ax, method) in enumerate(zip(axes[1:], methods)):
        log_probs_method = logprobs_chains[method].flatten()
        ax.hist(
            log_probs_method,
            bins=50,
            alpha=0.5,
            color=colors[i],
            density=True,
        )
        ax.set_title(method)
        ax.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
        gmm_best_sampler = log_probs_method[np.argmax(log_probs_method)]
        print(f"best {method} log prob: {gmm_best_sampler}")
        ax.axvline(
            gmm_best_sampler,
            lw=2,
            color=colors[i],
            ls="--",
        )
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(f"{savedir}/protein_reg3d3d_hist_logprobs.png", dpi=150)


def plot_avg_sampler_run_times(log_folder, methods, return_times=False):
    file_pattern = "reg_protein_3d3d_samples_chain"

    # Initialize dictionaries to store times
    sampler_times = {"sss-reject": [], "sss-shrink": [], "rwmh": [], "hmc": []}

    # Process each log file
    for i in range(1000):  # Assuming the files are named from 0 to 999
        log_file = os.path.join(log_folder, f"{file_pattern}{i}.log")
        if not os.path.exists(log_file):
            continue

        with open(log_file, "r") as f:
            content = f.read()
            # Extract sampler times using regex
            for sampler in sampler_times.keys():
                match = re.search(rf"{sampler} took ([\d.]+) s", content)
                if match:
                    sampler_times[sampler].append(float(match.group(1)))

    # Compute average times
    average_times = {
        sampler: np.mean(times) if times else 0
        for sampler, times in sampler_times.items()
    }

    if return_times:
        return list(average_times.values())

    # Print results
    print("Average times for each sampler:")
    for sampler, avg_time in average_times.items():
        print(f"{sampler}: {avg_time:.2f} s")
    # Plot the average times for each sampler
    fig, ax = plt.subplots(figsize=(10, 6))
    times = list(average_times.values())
    colors = ["tab:blue", "orange", "tab:green", "tab:red"]
    ax.bar(methods, times, color=colors)
    ax.set_title("Average sampler times for 2000 samples")
    ax.set_xlabel("MCMC sampler")
    ax.set_ylabel("Average time (s)")
    fig.tight_layout()

    fig.savefig(f"{log_folder}/protein_reg3d3d_run_times.png", dpi=200)


def compute_and_plot_sampler_success(
    logprobs_chains, methods, savedir, criteria, n_samples
):
    # create success criteria (5% of max log prob)

    print(f"success criteria {criteria: .2f}")
    print("=============")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    n_step = 10  # every 10 samples
    n_complete = (n_samples // n_step) * n_step
    for method in methods:
        # computing mean for every consecutive 10 samples
        samples_truncated = logprobs_chains[method][:, :n_complete]
        samples_mean_every_nstep = np.mean(
            samples_truncated.reshape(-1, n_samples // n_step, n_step), axis=2
        )

        # samples greater than the threshold
        success_rate = samples_mean_every_nstep > criteria
        success_rate_over_chains = np.mean(success_rate, axis=0)
        print(f"Success rate for {method}: {success_rate_over_chains}")
        x = np.linspace(0, n_samples, n_samples // n_step)
        y = success_rate_over_chains * 100
        ax.scatter(
            x,
            y,
            s=10,
            marker="o",
            label=method,
        )
        ax.plot(x, y, ls="--")
        ax.set_title("Success rate of samples")
        ax.set_xlabel("MCMC iterations")
        ax.set_ylabel("success rate [%]")
        ax.legend()

    fig.savefig(f"{savedir}/protein_reg3d3d_success_rate.png", dpi=200)


def compute_and_plot_sampler_success_times(
    logprobs_chains,
    methods,
    savedir,
    criteria,
    n_samples,
):
    # create success criteria (5% of max log prob)

    print(f"success criteria {criteria: .2f}")
    print("=============")
    times = plot_avg_sampler_run_times(savedir, methods, return_times=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    n_step = 10  # every 10 samples
    n_complete = (n_samples // n_step) * n_step
    for i, method in enumerate(methods):
        # computing mean for every consecutive 10 samples
        logprobs_truncated = logprobs_chains[method][:, :n_complete]
        logprobs_mean_every_nstep = np.mean(
            logprobs_truncated.reshape(-1, n_samples // n_step, n_step), axis=2
        )

        # samples greater than the threshold
        success_rate = logprobs_mean_every_nstep > criteria
        success_rate_over_chains = np.mean(success_rate, axis=0)
        print(f"Success rate for {method}: {success_rate_over_chains}")
        x = np.linspace(0, 1, n_samples // n_step) * times[i]
        y = success_rate_over_chains * 100
        ax.scatter(
            x,
            y,
            s=10,
            marker="o",
            label=method,
        )
        ax.plot(x, y, ls="--")
        ax.set_title("Success rate of samples")
        ax.set_xlabel("Computation time [s]")
        ax.set_ylabel("success rate [%]")
        ax.legend()
        ax.set_xlim(0, times[1])

    plt.show()

    fig.savefig(f"{savedir}/protein_reg3d3d_success_rate_times.png", dpi=200)


if __name__ == "__main__":
    METHODS = ("sss-reject", "sss-shrink", "rwmh", "hmc")
    data = np.load("data/protein_registration.npz")

    args = argparser()

    # Sampler Parameters
    n_samples = args["n_samples"]
    burnin = args["burnin"]
    n_chains = args["n_chains"]
    n_jobs = args["n_jobs"]
    reprod_switch = True

    # Directory to save results
    if args["out_dir"] is not None:
        savedir = args["out_dir"]
    else:
        savedir = f"results/protein_reg3d3d_CPD_chains_{n_chains}/"
    os.makedirs(savedir, exist_ok=True)
    hdf5_filepath = f"{savedir}/reg_protein_3d3d_samples.hdf5"

    # Set up logging
    logpath = f"{savedir}/parameters.log"
    setup_logging(logpath)

    # model parameters
    target = PointCloud(
        positions=data["target"],
        weights=None,
    )

    n_points = len(data["target"])
    source_weights = np.full(n_points, 1 / n_points)
    source = PointCloud(
        positions=data["source"],
        weights=source_weights,
    )

    sigma = data["sigma"]
    omega = data["prob_outlier"]

    # Log messages for important parameters
    logging.info(
        "\n---------------------------------\n"
        "Scoring metric: CPD\n"
        f"Number of samples: {n_samples}\n"
        f"Burn-in fraction: {burnin}\n"
        f"Number of chains: {n_chains}\n"
        f"Number of jobs: {n_jobs}\n"
        f"Output directory: {savedir}\n"
        "---------------------------------\n"
    )

    # create the target pdf with CPD
    target_pdf = CoherentPointDrift(
        target,
        source,
        sigma=sigma,
        k=20,
        omega=omega,
    )

    # sampling for chains in parallel
    rerun_samplers = False  # Set to True to always rerun samplers
    if not rerun_samplers and os.path.exists(hdf5_filepath):
        logging.info(f"File {hdf5_filepath} already exists. Loading existing data.")
        with h5py.File(hdf5_filepath, "r") as hf:
            # Load existing samples and logprobs
            samples_chains = {method: hf["samples"][method][()] for method in METHODS}
            logprobs_chains = {method: hf["logprobs"][method][()] for method in METHODS}
    else:
        logging.info(f"File {hdf5_filepath} does not exist. Starting sampling.")
        # Launch sampling for all chains
        # This will run the samplers in parallel and save results to HDF5
        samples_chains, logprobs_chains = launch_sampling_nchains(
            METHODS,
            target_pdf,
            n_samples,
            burnin,
            hdf5_filepath,
            reprod_switch,
            n_chains,
            seed_sequence=48385,
            return_chains=True,
            n_jobs=n_jobs,
            return_all_samples=False,  # extra `int(burnin * n_samples)` samples will be returned,
            stepsize_rwmh=0.015,  # Fixed (optimal) value, for automatic tuning, set burnin > 0.0
            stepsize_hmc=0.012,  # Fixed (optimal) value, for automatic tuning, set burnin > 0.0
        )

    # systematic rotational search via 600-cell for generating ground truth
    if not os.path.exists(f"{savedir}/log_probs_tesselations.hdf5"):
        # Generate rotations for systematic search via 600-cell
        print("Generating rotations for systematic search...")
        quaternions_tesselations = tessellate_rotations(n_discretize=3)
        print(f"Generated {len(quaternions_tesselations)} rotations")

        # Evaluate rotations
        print("Evaluating rotations with 600-cell...")
        log_probs_tesselations = np.zeros(len(quaternions_tesselations))
        for i, q in enumerate(quaternions_tesselations):
            # log_probs[i] = target_pdf._log_prob_R(q)
            log_probs_tesselations[i] = target_pdf.log_prob(q)
            if i % 1000 == 0:  # Show progress every 1000 rotations
                print(f"Evaluated {i}/{len(quaternions_tesselations)} rotations")

        with h5py.File(f"{savedir}/log_probs_tesselations.hdf5", "w") as hf:
            hf.create_dataset("quaternions", data=quaternions_tesselations)
            hf.create_dataset("log_probs", data=log_probs_tesselations)
    else:
        with h5py.File(f"{savedir}/log_probs_tesselations.hdf5", "r") as hf:
            quaternions_tesselations = hf["quaternions"][()]
            log_probs_tesselations = hf["log_probs"][()]

    # get the best log prob from tesselations and plot sampler success rate
    # TODO: Rerun when sampler params are decided to compute success rate
    best_log_prob = log_probs_tesselations.max()
    criteria_threshold = 0.05
    criteria = best_log_prob + criteria_threshold * best_log_prob

    # manually set the criteria for success rate
    # and plot histograms of log probabilities
    criteria = -2300
    plot_hist_logprobs(
        log_probs_tesselations,
        logprobs_chains,
        METHODS,
        savedir,
        criteria,
    )

    compute_and_plot_sampler_success_times(
        logprobs_chains,
        METHODS,
        savedir,
        criteria,
        n_samples,
    )

    # NOTE: `n_samples` will not use the last `int(burnin * n_samples)` samples,
    # if needed, this can be explicitly added to `n_samples`.
    # See `return_all_samples=True`
    compute_and_plot_sampler_success(
        logprobs_chains,
        METHODS,
        savedir,
        criteria,
        n_samples,
    )

    # plot diagnostics for samplers and store results
    plot_heatmap_logprobs(logprobs_chains, METHODS, savedir)
    plot_avg_sampler_run_times(savedir, METHODS)
