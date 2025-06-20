import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from geosss.io import load

METHODS = ["sss-reject", "sss-shrink", "rwmh", "hmc"]
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
}

plt.rc("font", size=11)


def get_dataset(d, K, path, kappas):
    # create a list[dict] for values
    datasets = []
    for kappa in kappas:
        subdir = f"mixture_vMF_d{d}_K{K}_kappa{kappa}"
        ess_file = f"{path}/{subdir}/{subdir}_ess.pkl.gz"
        ess = load(ess_file, gzip=True)
        for method in METHODS:
            for ess_val in ess[method]:
                datasets.append(
                    {"Kappa": kappa, "Method": ALGOS[method], "ESS": ess_val}
                )

    return pd.DataFrame(datasets)


def ess_boxplot(d, K, kappas, path, savefig=True):
    """
    Box plot for ess values per dimension for varying kappas.


    Args:
        d (int): dimension
        K (int): components
        kappas (float): concentration parameter
        path (str): results dir
        savefig (bool): saving figure
    """

    algos = ALGOS

    # convert the extracted pandas dataframe
    ess_df = get_dataset(d, K, path, kappas)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # sns box plot
    sns.boxplot(
        data=ess_df,
        x="Kappa",
        y="ESS",
        hue="Method",
        hue_order=list(algos.values()),
        palette="Set2",
        ax=ax,
    )

    # add labels and titles
    ax.set_yscale("log")
    ax.set_ylabel("relative ESS (Log scale)")
    ax.legend()
    plt.xticks()

    # add vertical lines between the Kappa values
    for i in range(len(kappas)):
        x = 0.5 + i
        ax.axvline(x=x, linestyle="--", color="gray", alpha=0.8)

    if savefig:
        savefile = (
            f"{path}/d{d}_K{K}_kappa{kappas.min()}_{kappas.max()}_ess_boxplot.pdf"
        )
        fig.savefig(savefile, transparent=True)


def extract_evals(d, K, kappas, path) -> dict:
    """
    Extracting the evaluations for log_prob, gradient, rejected samples
    from every log txt file (consisting of 10 runs) that corresponds to a
    specific kappa.

    Ex: For extracting logprob values of sss-reject for a kappa of 100
    vals = evals[100]['sss-reject]['logprob']

    Args:
        d (int): dimension
        K (int, optional): No. of components of mixture of vMF. Defaults to 5.
        kappas (ndarray or list, optional): list of concentration parameters. Defaults to None.

    Returns:
        evals (dict[dict[list]])
    """

    if kappas is None:
        kappas = np.arange(50, 550, 50)

    methods = ["sss-reject", "sss-shrink", "rwmh", "hmc"]

    evals = {kappa: None for kappa in kappas}

    for kappa in kappas:
        # load file path
        subdir = f"mixture_vMF_d{d}_K{K}_kappa{kappa}"
        filepath = f"{path}/{subdir}/{subdir}_log.txt"

        # read and extract details from the txt file
        with open(f"{filepath}", "r") as file:
            # create an empty dict[dict[list]]
            evals_kappa = {
                method: {"logprob": [], "reject": [], "grad": []} for method in methods
            }

            for line in file:
                # search for logprob through all methods
                for method in methods:
                    match_logprob = re.search(
                        rf"logprob calls for {method}: (\d+)", line
                    )
                    if match_logprob:
                        logprob_num = int(match_logprob.group(1))
                        evals_kappa[method]["logprob"].append(logprob_num)

                # search for rejected nums through slice samplers
                for method in ["sss-reject", "sss-shrink"]:
                    match_reject = re.search(
                        rf"Rejected samples for {method}: (\d+)", line
                    )
                    if match_reject:
                        reject_num = int(match_reject.group(1))
                        evals_kappa[method]["reject"].append(reject_num)

                # gradient evals for hmc
                match_grad = re.search(r"gradient calls for hmc: (\d+)", line)
                if match_grad:
                    grad_num = int(match_grad.group(1))
                    evals_kappa["hmc"]["grad"].append(grad_num)

        # update dictionary for the file corresponding to `kappa`
        evals[kappa] = evals_kappa

    return evals


def ess_plot(
    d: int,
    K: int,
    path: str,
    kappas: np.ndarray,
    dim: int = 0,
    ax=None,
) -> None:
    """
    Plotting ess values for a given dimension `dim` amongst `d` dimensions against kappa
    for all the samplers.

    Args:
        d (int): dimension of the mixture of vMF
        K (int): mixture components
        kappas (float): concentration parameter
        path (str): results dir
        dim (int, optional): selecting a dimension to plot. Defaults to 0.
        savefig (bool, optional): save figure. defaults to True

    """

    # Create figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # convert the extracted pandas dataframe
    ess_df = get_dataset(d, K, path, kappas)

    # extract single value for the given `dim` dimension (default=0)
    ess_vals = {method: [] for method in METHODS}
    for method in METHODS:
        for kappa in kappas:
            ess_val = ess_df.loc[
                (ess_df["Kappa"] == kappa) & (ess_df["Method"] == ALGOS[method]), "ESS"
            ].iloc[dim]
            ess_vals[method].append(ess_val)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    markers = ["8", "s", "^", "P"]
    color_palette = sns.color_palette("deep", n_colors=len(METHODS))
    for i, method in enumerate(METHODS):
        label = ALGOS[method]
        ax.plot(
            kappas,
            ess_vals[method],
            marker=markers[i],
            markersize=6,
            label=label,
            color=color_palette[i],
        )
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel(r"concentration parameter $\kappa$")
    ax.set_ylabel("relative ESS (log)")

    # Set the x-tick locations and labels
    ax.set_xticks(kappas)
    ax.set_xticklabels(kappas)

    return fig if standalone else ax


def plot_rejected_samples(d, K, kappas, path, ax=None):
    """Plotting the number of rejected samples for slice samplers"""

    # Create figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    methods = METHODS[:2]
    evals = extract_evals(d, K, kappas, path)

    # extract values from a single run for different kappas
    vals_shrink = []
    vals_reject = []
    for kappa in kappas:
        # normalized over 1e6 samples
        vals_reject.append(evals[kappa]["sss-reject"]["reject"][0] / 1e6)
        vals_shrink.append(evals[kappa]["sss-shrink"]["reject"][0] / 1e6)

    print(vals_reject)
    print(vals_shrink)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    markers = ["8", "s"]
    color_palette = sns.color_palette("deep", 2)
    for i, vals in enumerate([vals_reject, vals_shrink]):
        ax.plot(
            kappas,
            vals,
            marker=markers[i],
            markersize=6,
            label=ALGOS[methods[i]],
            color=color_palette[i],
        )
    ax.set_xlabel(r"concentration parameter $\kappa$")
    ax.set_ylabel("number of rejections")
    # ax.legend()

    # Set the x-tick locations and labels
    ax.set_xticks(kappas)
    ax.set_xticklabels(kappas)

    return fig if standalone else ax


if __name__ == "__main__":
    # params
    d = 10
    K = 5
    kappas = np.arange(50, 550, 50)
    path = f"results/mix_vMF_d{d}_K{K}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # assumes precomputed ess for kappa values between 50 and 500
    ess_plot(d, K, path, kappas, ax=ax1)
    ax1.set_ylim(None, 1e-2)

    plot_rejected_samples(d, K, kappas, path, ax=ax2)

    for i, ax in enumerate((ax1, ax2), 65):
        ax.annotate(chr(i), xy=(0.02, 0.91), xycoords="axes fraction", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        f"{path}/mixture_vMF_d{d}_K{K}_kappa50_500_ess_nrejections_plot.pdf",
        transparent=True,
        dpi=200,
    )

    plt.show()
