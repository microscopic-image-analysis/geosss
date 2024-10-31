# just a parallel implementation that runs the 4 algos on 4 cpus for vMF mixture

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from functools import partial

import numpy as np
from csb.io import dump, load

import geosss as gs
import geosss.vMF_diagnostics as vis


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
    n_runs: int = 1,
    reprod_switch: bool = True,
):
    """Run all the samplers"""

    # generate fixed seeds based on `n_runs`
    if reprod_switch:
        ss = np.random.SeedSequence(48385)
        seeds = ss.spawn(n_runs)

    # start samplers
    runs_samples = []
    runs_logprob = []
    for i in range(n_runs):
        print(f"\nRun {i+1}\n-------------------------------")

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


def load_or_run(pkl_path, pdf, methods, n_samples, burnin, n_runs, reprod_switch):
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
                n_runs=n_runs,
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

    # load ess results per dimension or compute if n_runs = 10
    vis.calc_ess(runs_samples, methods, pkl_path, return_ess=False)

    return runs_samples


def cli_args(d, K, kappa, n_samples, n_runs):
    """
    command-line interface for the given arguments
    """

    # parser description
    parser = argparse.ArgumentParser(
        description="Loading dimension (d), Component (K) and concentration parameter (kappa)"
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
        "--component",
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
        "-n_samples",
        "--n_samples",
        required=False,
        default=n_samples,
        help="no. of samples",
        type=int,
    )
    parser.add_argument(
        "-n_runs",
        "--n_runs",
        required=False,
        default=n_runs,
        help="no. of runs per sampler",
        type=int,
    )
    parser.add_argument(
        "-o", "--out_dir", required=False, help="main output directory", default="./"
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

    vis.acf_kld_dist_plot(
        samples, pdf, path, filename, lag=acf_lag, fs=16, save_res=save_res
    )

    # plot histogram with mixture of true vMF marginal
    vis.hist_plot_mixture_marginals(
        pdf, samples, ndim, path, filename, save_res=save_res
    )

    # additional-plots (not used in paper)
    if misc_plots:

        # geodesic distance
        vis.dist_plot(samples, pdf, kappa, path, filename, save_res=save_res)

        # ACF and entropy plots
        vis.acf_entropy_plot(
            samples, pdf, path, filename, lag=acf_lag, save_res=save_res
        )

        # entropy and kl-divergence plots
        vis.entropy_kld(samples, pdf, path, filename, save_res=save_res)

        # trace plots per dimension
        vis.trace_plots(samples, ndim, path, filename, save_res=save_res)

        # ACF plot per dimension
        vis.acf_plots(samples, ndim, path, filename, lag=acf_lag, save_res=save_res)


def main():
    # set the parameters
    d = 10  # dimension
    K = 5  # number of mixture components
    kappa = 100.0  # concentration parameter
    reprod_switch = True  # generates reproducible results
    plot_results = True  # plotting results
    save_results = True  # saving results
    n_samples = int(1e6)  # no. of samples
    n_runs = 1  # sampler runs (ess only for `n_runs=10`)
    burnin = int(0.1 * n_samples)  # burnin samples

    # set filepaths and filenames
    PATH = f"results/mix_vMF_d{d}_K{K}"
    filename = f"mixture_vMF_d{d}_K{K}_kappa{int(kappa)}"
    subdir = os.path.join(PATH, filename)

    # uses the above params as default for cli args
    args = cli_args(d, K, kappa, n_samples, n_runs)

    # modified from console
    d = args["dimension"]
    K = args["component"]
    kappa = args["concentration"]
    n_samples = args["n_samples"]
    n_runs = args["n_runs"]

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
        f"{subdir}/{filename}", pdf, methods, n_samples, burnin, n_runs, reprod_switch
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
