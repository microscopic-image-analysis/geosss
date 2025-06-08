import os

import arviz as az
import numpy as np
from geosss.io import dump, load

import geosss as gs


class SamplerLauncher(gs.SamplerLauncher):
    """Adding the kent method to the sampler interface to generate
    ground truth samples
    """

    def run_kent(self):
        return gs.sample_bingham(self.pdf.A, self.n_samples)

    def run(self, method):
        return self.run_kent() if method == "kent" else super().run(method)


def hopping_frequency(samples, pdf):
    """hopping frequency between the two modes of bingham"""
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
                print("Calculating ESS from samples..")
                ess = {method: None for method in methods}

                for method in methods:
                    # projects samples to the mode from all dimensions
                    samples = np.array(
                        [draws[method] @ pdf.mode for draws in runs_samples]
                    )

                    # estimates ESS using the arviz library per dimension
                    samples_az = az.convert_to_dataset(samples)
                    ess_val = az.ess(samples_az, relative=True)
                    ess[method] = ess_val.x.values

                print(f"Saving ESS file {ess_file}")
                dump(ess, ess_file, gzip=True)

            else:
                print("ESS values not computed, requires `n_runs=10`")
                return None

    for method in methods:
        print(f"ESS for {method}: {ess[method]:.3%}")

    if return_ess:
        return ess


def launch_samplers(
    savedir, d, vmax, pdf, initial, n_samples, burnin, methods, reprod_switch
):
    # load samples
    try:
        runs_samples = load(f"{savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")
        print(f"Loading file {savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")

    # run samplers
    except FileNotFoundError:
        print("File not found, starting samplers..")

        # generate fixed seeds based on `n_runs`
        if reprod_switch:
            ss = np.random.SeedSequence(48385)
            seeds = ss.spawn(n_runs)

        runs_samples = []
        for i in range(n_runs):
            # tester is instantiated based on seed
            seed = seeds[i] if reprod_switch else None
            launcher = SamplerLauncher(pdf, initial, n_samples, burnin, seed=seed)

            # samples saved as dict
            samples = {}
            samples["kent"] = launcher.run("kent")
            for method in methods:
                with gs.take_time(method):
                    samples[method] = launcher.run(method)

            # append for every run
            runs_samples.append(samples)

        # save a copy
        dump(runs_samples, f"{savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")
        print(f"Saving file {savedir}/bingham_d{d}_vmax{int(vmax)}.pkl")

    return runs_samples


if __name__ == "__main__":
    n_samples = int(1e5)  # number of samples
    burnin = int(0.1 * n_samples)  # burn-in
    n_runs = 10  # no. of runs (ESS for `n_runs=10`)
    reprod_switch = True  # make the samplers reproducible
    save_figs = True  # save plots

    # dimension and lambda, ind 0 or 1 for plots with
    # 10 and 50 dimensions respectively
    ind = 0
    d, vmax = [(10, 30.0), (50, 300.0)][ind]

    # save directory (generate results/ dir if non-existent)
    filename = f"bingham_d{d}_vmax{int(vmax)}"
    savedir = f"results/{filename}"
    os.makedirs(savedir, exist_ok=True)

    # bingham distribution as pdf which is fixed
    pdf = gs.random_bingham(d=d, vmax=vmax, vmin=0.0, eigensystem=True, seed=6982)

    # sampler methods
    initial = pdf.mode
    methods = ("sss-reject", "sss-shrink", "rwmh", "hmc")
    algos = {
        "sss-reject": "geoSSS (reject)",
        "sss-shrink": "geoSSS (shrink)",
        "rwmh": "RWMH",
        "hmc": "HMC",
    }

    # launch samplers with initial state at the mode of pdf
    runs_samples = launch_samplers(
        savedir, d, vmax, pdf, initial, n_samples, burnin, methods, reprod_switch
    )

    # calculate ess if `n_runs=10`
    bingham_ess(runs_samples, pdf, methods, f"{savedir}/{filename}", return_ess=False)

    # Loading the first run `ind=0` to generate plots in paper
    ind = 0
    samples = runs_samples if isinstance(runs_samples, dict) else runs_samples[ind]

    # NOTE: For plotting results and conclusions, see `Bingham.ipynb`
