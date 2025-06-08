# just a parallel implementation that runs the 4 algos on 4 cpus for vMF mixture

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from functools import partial

import numpy as np

import geosss as gs
from geosss.io import dump, load
from scripts.vMF_diagnostics import (
    acf_entropy_plot,
    acf_kld_dist_plot,
    acf_plots,
    calc_ess,
    dist_plot,
    entropy_kld,
    hist_plot_mixture_marginals,
    trace_plots,
)


def _rotate_north(u):
    """
    Find rotation of u onto north pole [0 ... 0 1].
    """
    v = np.zeros_like(u)
    v[-1] = 1

    U, _, V = np.linalg.svd(np.multiply.outer(v, u))

    R = U @ V

    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1
        R = U @ V

    return R


def sample_vMF(pdf, size=1):
    """
    Generates a random sample from the von Mises-Fisher distribution using the
    algorithm proposed by Wood (1994).
    """
    assert isinstance(pdf, gs.VonMisesFisher)

    if size > 1:
        return np.array([sample_vMF(pdf) for _ in range(int(size))])

    p = pdf.d - 1
    kappa = np.linalg.norm(pdf.mu)

    if np.isclose(kappa, 0.0):
        return gs.sample_sphere(p)

    u = pdf.mu / kappa

    b0 = (-2 * kappa + (4 * kappa**2 + p**2) ** 0.5) / p
    x0 = (1 - b0) / (1 + b0)
    n = kappa * x0 + p * np.log(1 - x0**2)

    while True:
        Z = np.random.beta(0.5 * p, 0.5 * p)
        U = np.log(np.random.rand())
        W = (1 - (1 + b0) * Z) / (1 - (1 - b0) * Z)

        if (kappa * W + p * np.log(1 - x0 * W) - n) >= U:
            break

    # sample from d-2 sphere
    v = gs.sample_sphere(p - 1)
    x = np.append((1 - W**2) ** 0.5 * v, W)
    R = _rotate_north(u).T

    return R @ x


class SamplerLauncher(gs.SamplerLauncher):
    def run_wood(self):
        N = np.random.multinomial(self.n_samples, self.pdf.weights)
        samples = [gs.sample_vMF(pdf, n) for pdf, n in zip(self.pdf.pdfs, N)]
        samples = np.vstack(samples)
        return samples[np.random.permutation(self.n_samples)]

    def run(self, method):
        return self.run_wood() if method == "wood" else super().run(method)


def sampler(method: str, launcher: SamplerLauncher) -> dict[str]:
    """calls a sampling method"""

    # run sampler for a given method
    samples = {}
    logprob = {}

    pdf = launcher.pdf
    with gs.take_time(method):
        # sampler for the given method and also save logprob
        samples[method] = launcher.run(method)
        logprob[method] = pdf.log_prob(samples[method])

        # gradient/log_prob calls
        print(f"gradient calls for {method}:", pdf.gradient.num_calls)
        print(f"logprob calls for {method}:", pdf.log_prob.num_calls)

        # counter for rejected samples
        if method == "sss-reject":
            print(f"Rejected samples for {method}: {launcher.rsss.n_reject}")

        if method == "sss-shrink":
            print(f"Rejected samples for {method}: {launcher.ssss.n_reject}")

        # reset counters just in case (it doesn't reset if it is not a
        # parallel implementation)
        pdf.log_prob.reset_counters()
        pdf.gradient.reset_counters()

    print("-------")

    return samples, logprob


def run_samplers(
    pdf,
    methods: list[str],
    n_samples: int,
    burnin: int,
    n_chains: int = 1,
    reprod_switch: bool = True,
):
    """Run all the samplers"""

    # generate fixed seeds based on `n_chains`
    if reprod_switch:
        ss = np.random.SeedSequence(48385)
        seeds = ss.spawn(n_chains)

    # start samplers
    runs_samples = []
    runs_logprob = []
    for i in range(n_chains):
        print(f"\nRun {i + 1}\n-------------------------------")

        # fixes seed for initial state and samplers
        seed = seeds[i] if reprod_switch else None
        init_state = gs.sample_sphere(pdf.pdfs[0].d - 1, seed=seed)
        print(f"initial state: {init_state}")

        # tester that starts samplers
        tester = SamplerLauncher(pdf, init_state, n_samples, burnin, seed)

        # initialize samples dict and load the wood samples
        samples = {method: None for method in methods}
        samples["wood"] = tester.run("wood")

        # initialize logprob
        logprob = {method: None for method in methods}
        logprob["wood"] = tester.pdf.log_prob(samples["wood"])

        # create a partial function
        sampler_partial = partial(sampler, tester=tester)

        # Run all the samplers in parallel
        with ProcessPoolExecutor(max_workers=len(methods)) as exe:
            futures = exe.map(sampler_partial, methods)

        # new_samples = list(res)
        results = list(futures)

        # unpack the results for each method
        for i in range(len(methods)):
            new_sample, new_logprob = results[i]
            samples.update(new_sample)
            logprob.update(new_logprob)

        # merge runs
        runs_samples.append(samples)
        runs_logprob.append(logprob)

    return runs_samples, runs_logprob


def load_or_run(
    pkl_path,
    pdf,
    methods,
    n_samples,
    burnin,
    n_chains,
    reprod_switch,
):
    """Loads the samples from memory or runs the sampler"""

    pklfile_samples = f"{pkl_path}.pkl.gz"
    pklfile_logprob = f"{pkl_path}_logprob.pkl.gz"

    # load samples and logprob
    try:
        runs_samples = load(pklfile_samples, gzip=True)
        print(f"Loading file {pklfile_samples}")

    # run samplers and save the result
    except FileNotFoundError:
        print("File not found, starting samplers..")

        def start_samplers():
            """convenience function"""

            # start the samplers parallely
            start = time.perf_counter()
            runs = run_samplers(
                pdf,
                methods,
                n_samples,
                burnin,
                n_chains=n_chains,
                reprod_switch=reprod_switch,
            )
            runs_samples, runs_logprob = runs
            end = time.perf_counter()

            print(f"\nTotal time elapsed: {end - start:.1f} s")

            # save the samples runs
            dump(runs_samples, pklfile_samples, gzip=True)
            print(f"Saving file {pklfile_samples}")

            # save the logprob runs
            dump(runs_logprob, pklfile_logprob, gzip=True)
            print(f"Saving file {pklfile_logprob}")

            return runs_samples

        # save the print output to a log file
        with open(f"{pkl_path}_log.txt", "w") as f:
            with redirect_stdout(f):
                runs_samples = start_samplers()

    # load ess results per dimension or compute if n_chains = 10
    calc_ess(runs_samples, methods, pkl_path, return_ess=False)

    return runs_samples


def cli_args(d, K, kappa, n_samples, n_chains):
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


def visualize_samples(
    samples, kappa, pdf, path, filename, save_res=True, misc_plots=False, acf_lag=80000
):
    """Just a util routine that calls all visualizing functions"""

    # modes of a mixture model
    ndim = pdf.pdfs[0].d

    acf_kld_dist_plot(
        samples,
        pdf,
        path,
        filename,
        lag=acf_lag,
        fs=16,
        save_res=save_res,
    )

    # plot histogram with mixture of true vMF marginal
    hist_plot_mixture_marginals(
        pdf,
        samples,
        ndim,
        path,
        filename,
        save_res=save_res,
    )

    # additional-plots (not used in paper)
    if misc_plots:
        # geodesic distance
        dist_plot(samples, pdf, kappa, path, filename, save_res=save_res)

        # ACF and entropy plots
        acf_entropy_plot(samples, pdf, path, filename, lag=acf_lag, save_res=save_res)

        # entropy and kl-divergence plots
        entropy_kld(samples, pdf, path, filename, save_res=save_res)

        # trace plots per dimension
        trace_plots(samples, ndim, path, filename, save_res=save_res)

        # ACF plot per dimension
        acf_plots(samples, ndim, path, filename, lag=acf_lag, save_res=save_res)


def main():
    # set the parameters
    d = 10  # dimension
    K = 5  # number of mixture components
    kappa = 100.0  # concentration parameter
    reprod_switch = True  # generates reproducible results
    plot_results = True  # plotting results
    save_results = True  # saving results
    n_samples = int(1e6)  # no. of samples
    n_chains = 1  # sampler runs (ess only for `n_chains=10`)
    burnin = int(0.1 * n_samples)  # burnin samples

    # set filepaths and filenames
    PATH = f"results/mix_vMF_d{d}_K{K}"
    filename = f"mixture_vMF_d{d}_K{K}_kappa{int(kappa)}"
    subdir = os.path.join(PATH, filename)

    # uses the above params as default for cli args
    args = cli_args(d, K, kappa, n_samples, n_chains)

    # modified from console
    n_samples = args["n_samples"]
    burnin = args["burnin"]
    n_chains = args["n_chains"]
    d = args["dimension"]
    K = args["components"]
    kappa = args["concentration"]

    # update the path if arg specified as command line
    subdir = args["out_dir"] + subdir

    # create the subdir if it doesn't exist
    os.makedirs(subdir, exist_ok=True)

    # fixes modes to fix the target
    mode_seed = 1234
    modes = gs.sphere.sample_sphere(d - 1, K, seed=mode_seed)

    # pdf as a mixture of von Mises-Fisher distributions
    vmfs = [gs.VonMisesFisher(kappa * mu) for mu in modes]
    pdf = gs.MixtureModel(vmfs)

    # sampler methods
    methods = ("sss-reject", "sss-shrink", "rwmh", "hmc")

    # load samples or run sampler
    runs_samples = load_or_run(
        f"{subdir}/{filename}", pdf, methods, n_samples, burnin, n_chains, reprod_switch
    )

    # Loading the first run `ind=0` to generate plots in paper
    ind = 0
    samples = runs_samples if isinstance(runs_samples, dict) else runs_samples[ind]

    # plotting results
    if plot_results:
        print("Plotting results..")
        visualize_samples(
            samples,
            kappa,
            pdf,
            subdir,
            filename,
            save_res=save_results,
            misc_plots=False,  # set `true` for misc plots not in paper
            acf_lag=int(8e4),  # adjusting `acf_lag` if not sufficient
        )


if __name__ == "__main__":
    main()
