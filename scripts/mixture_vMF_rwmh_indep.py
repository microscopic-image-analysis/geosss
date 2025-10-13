# Runs the mixture sampler `mcmc.MixtureRWMHIndependenceSampler` on the vMF mixture for the fixed parameters:
# d=10, K=5, n_chains=10, n_samples=1e6 and burnin=0.2.

# The concentration parameter kappa is sweeped from 50 to 500 in steps of 50 and mixing probability from 0.1 to 1.0
# in steps of 0.1 via the SLURM script sh/submit_job_mixture_sampler_vMF_varying_kappa.sh.

# The results are saved in results/mix_vMF_d10_K5/mixture_vMF_d10_K5_kappa<kappa>/sampler_mixture_rwmh_indep

import argparse
import logging
import os
import time

import numpy as np
from joblib import Parallel, delayed

import geosss as gs
from geosss.io import dump
from geosss.mcmc import MixtureRWMHIndependenceSampler


def setup_logging(logpath: str, filemode: str = "w"):
    """Setting up logging

    Parameters
    ----------
    logpath : str
        full path to log file
    filemode : str
        'w' to overwrite the log file, 'a' to append (default: 'w')
    """

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=logpath,
        filemode=filemode,
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def _sampler_single_chain(
    pdf,
    n_samples: int,
    burnin: int,
    mixing_prob: float,
    seed=None,
):
    """Run a single chain of MixtureRWMHIndependenceSampler and return samples + statistics."""

    # Start timing
    start_time = time.time()

    # Generate initial state
    init_state = gs.sample_sphere(pdf.pdfs[0].d - 1, seed=seed)
    logging.info(f"Initial state for this chain: {init_state}")

    # Create and run sampler
    sampler = MixtureRWMHIndependenceSampler(
        pdf,
        init_state,
        seed,
        stepsize=0.1,
        mixing_probability=mixing_prob,
    )

    logging.info(
        "\n---------------------------------------------------------------------\n"
        "Starting Mixture Sampler (RWMH + Independence)\n"
        "---------------------------------------------------------------------"
    )

    samples = sampler.sample(n_samples, burnin)

    # End timing
    elapsed_time = time.time() - start_time

    # Log statistics
    logging.info(
        "\n---------------------------------------------------------------------\n"
        "Sampling completed - Statistics\n"
        "---------------------------------------------------------------------"
    )
    # Collect statistics
    stats = {
        "init_state": init_state,
        "stepsize": sampler.stepsize,
        "rwmh_counter": sampler.rwmh_counter,
        "indep_counter": sampler.indep_counter,
        "n_accept": sampler.n_accept,
        "acceptance_rate": sampler.n_accept / (n_samples + burnin),
        "burnin_counter": sampler._counter,
        "elapsed_time": elapsed_time,
    }

    return {"samples": samples, "stats": stats}


def run_parallelized_chains(
    pdf,
    n_samples: int,
    burnin: int,
    n_chains: int = 10,
    mixing_prob: float = 0.1,
    reprod_switch: bool = True,
):
    """Run multiple chains of MixtureRWMHIndependenceSampler in parallel."""

    # Generate fixed seeds based on `n_chains`
    if reprod_switch:
        # NOTE: Same seed for comparison with other samplers (see scripts/mixture_vMF.py)
        ss = np.random.SeedSequence(48385)
        seeds = ss.spawn(n_chains)
    else:
        seeds = [None] * n_chains

    # Prepare arguments for each chain
    run_args = [
        (pdf, n_samples, burnin, mixing_prob, seeds[i]) for i in range(n_chains)
    ]

    # Execute chains in parallel
    if n_chains > 1:
        logging.info(f"Starting parallel sampling for {n_chains} chains...")
        results = Parallel(n_jobs=-1)(
            delayed(_sampler_single_chain)(*args) for args in run_args
        )
        logging.info("Parallel sampling completed.")
    else:
        # Single chain - execute directly without parallelization
        logging.info("Starting single chain sampling...")
        results = [_sampler_single_chain(*run_args[0])]
        logging.info("Single chain sampling completed.")

    # Aggregate results into a dictionary
    aggregated_results = {}
    logging.info("\n" + "=" * 70)
    logging.info("Summary of All Chains")
    logging.info("=" * 70)

    for i, result in enumerate(results):
        aggregated_results[f"run_{i}"] = result
        logging.info(f"\n=== Chain {i} Statistics ===")
        logging.info(f"Initial state: {result['stats']['init_state']}")
        logging.info(f"Final stepsize: {result['stats']['stepsize']:.6f}")
        logging.info(f"RWMH proposals: {result['stats']['rwmh_counter']}")
        logging.info(f"Independence proposals: {result['stats']['indep_counter']}")
        logging.info(f"Total acceptances: {result['stats']['n_accept']}")
        logging.info(f"Acceptance rate: {result['stats']['acceptance_rate']:.2%}")
        logging.info(f"Elapsed time: {result['stats']['elapsed_time']:.2f} seconds")

    return aggregated_results


def cli_args(d, K, kappa, mix_prob, n_samples, n_chains):
    """
    command-line interface for the given arguments
    """

    # parser description
    parser = argparse.ArgumentParser(
        description="Loading dimension (d), Component (K) and concentration parameter (kappa)"
    )

    parser.add_argument(
        "-n_samples",
        "--n_samples",
        required=False,
        default=n_samples,
        help="no. of samples",
        type=int,
    )

    parser.add_argument(
        "-burnin",
        "--burnin",
        required=False,
        default=0.1,
        help="fraction of burnin samples",
        type=float,
    )

    # parser args
    parser.add_argument(
        "-d",
        "--dimension",
        required=False,
        default=d,
        help="dimension of the vmf mixture",
        type=int,
    )

    parser.add_argument(
        "-K",
        "--components",
        required=False,
        default=K,
        help="no. of components of the mixture model",
        type=int,
    )

    parser.add_argument(
        "-kappa",
        "--concentration",
        required=False,
        default=kappa,
        help="concentration parameter of vMF",
        type=float,
    )

    parser.add_argument(
        "-mix_prob",
        "--mix_prob",
        required=False,
        default=mix_prob,
        help="mixture probability",
        type=float,
    )

    parser.add_argument(
        "-n_chains",
        "--n_chains",
        required=False,
        default=n_chains,
        help="no. of runs per sampler",
        type=int,
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        required=False,
        help="main output directory",
        default="./",
    )

    # load args
    args = vars(parser.parse_args())

    return args


def main():
    # set the parameters
    d = 10  # dimension
    K = 5  # number of mixture components
    kappa = 50.0  # concentration parameter
    reprod_switch = True  # generates reproducible results
    mix_prob = 0.1  # mixing probability
    n_samples = int(1e3)  # no. of samples
    n_chains = 10  # sampler runs (parallel chains)

    # uses the above params as default for cli args
    args = cli_args(d, K, kappa, mix_prob, n_samples, n_chains)

    # modified from console
    n_samples = args["n_samples"]
    burnin_fraction = args["burnin"]
    n_chains = args["n_chains"]
    d = args["dimension"]
    K = args["components"]
    kappa = args["concentration"]
    mix_prob = args["mix_prob"]
    out_dir = args["out_dir"]

    # Calculate burnin as integer
    burnin = int(burnin_fraction * n_samples)

    # Construct output directory path
    subdir = f"mix_vMF_d{d}_K{K}/mixture_vMF_d{d}_K{K}_kappa{int(kappa)}/sampler_mixture_rwmh_indep"
    if out_dir != "./":
        # If out_dir is provided, use it as base
        full_dir = os.path.join(out_dir, subdir)
    else:
        # Otherwise use results as base
        full_dir = os.path.join("results", subdir)

    # Create the directory if it doesn't exist
    os.makedirs(full_dir, exist_ok=True)

    # Set up main logging
    main_logpath = f"{full_dir}/mixture_sampler_kappa{kappa}_mixprob_{mix_prob}.log"
    setup_logging(main_logpath)

    # Log configuration
    logging.info("\n" + "=" * 70)
    logging.info("MixtureRWMHIndependenceSampler Configuration")
    logging.info("=" * 70)
    logging.info(f"Dimension: {d}")
    logging.info(f"Components: {K}")
    logging.info(f"Concentration (kappa): {kappa}")
    logging.info(f"Mixing Probability: {mix_prob}")
    logging.info(f"Number of samples: {n_samples}")
    logging.info(f"Burnin: {burnin} ({burnin_fraction * 100}%)")
    logging.info(f"Number of chains: {n_chains}")
    logging.info(f"Output directory: {full_dir}")
    logging.info(f"Reproducible: {reprod_switch}")
    logging.info("=" * 70 + "\n")

    # Fix modes to fix the target
    mode_seed = 1234
    modes = gs.sphere.sample_sphere(d - 1, K, seed=mode_seed)
    logging.info(f"Target modes (seed={mode_seed}):\n{modes}\n")

    # pdf as a mixture of von Mises-Fisher distributions
    vmfs = [gs.VonMisesFisher(kappa * mu) for mu in modes]
    pdf = gs.MixtureModel(vmfs)

    # Run the sampler for multiple chains
    results = run_parallelized_chains(
        pdf, n_samples, burnin, n_chains, mix_prob, reprod_switch
    )

    # Save results to a gzipped pickle file
    filename = f"sampler_mixture_rwmh_indep_kappa{kappa}_mixprob_{mix_prob}.pkl.gz"
    filepath = os.path.join(full_dir, filename)

    logging.info(f"\nSaving results to: {filepath}")
    dump(results, filepath, gzip=True)
    logging.info("Results saved successfully!")
    logging.info(f"Log file saved to: {main_logpath}")


if __name__ == "__main__":
    main()
