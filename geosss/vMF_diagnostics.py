# diagnostics and plots for mixture of vMF as target (although mostly for general targets,
# but need to check that)

import logging
import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from csb.io import dump, load

from geosss.distributions import MarginalVonMisesFisher, MixtureModel, VonMisesFisher
from geosss.sphere import distance
from geosss.utils import acf

METHODS = ["sss-reject", "sss-shrink", "rwmh", "hmc"]
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
}

plt.rc("font", size=16)


def hist_plot(samples, ndim, path, filename, fs=16, save_res=False):
    """
    histogram of samples.
    """
    bins = 100
    plt.rc("font", size=fs)

    # shows a standard histogram per dimension
    if ndim == 3:
        figsize = (10, 10)
    else:
        figsize = (10, 15)
    fig, rows = plt.subplots(ndim, len(METHODS), figsize=figsize, sharex=True)

    for i, axes in enumerate(rows):
        # reference samples
        wood_vals = samples["wood"][:, i]
        ref = list(np.histogram(wood_vals, bins=bins, density=True))
        ref[1] = 0.5 * (ref[1][1:] + ref[1][:-1])

        # show histogram
        for ax, method in zip(axes, METHODS):
            marginals = samples[method][:, i]

            bins = ax.hist(
                marginals,
                bins=bins,
                density=True,
                alpha=0.3,
                color="k",
                histtype="stepfilled",
            )[1]
            ax.plot(*ref[::-1], color="r", lw=1, ls="--")
            ax.set_xlabel(rf"$e_{i}^Tx_n$", fontsize=fs)

    for ax, method in zip(rows[0], METHODS):
        ax.set_title(ALGOS[method], fontsize=fs)
    fig.tight_layout()

    if save_res:
        print("Saving histogram plot..")
        fig.savefig(f"{path}/{filename}_hist.pdf", transparent=True)

    plt.close(fig)


def hist_plot_mixture_marginals(
    pdf, samples, ndim, path, filename, fs=16, save_res=False
):
    """
    histogram of samples.
    """
    bins = 100
    plt.rc("font", size=fs)

    # shows a standard histogram per dimension
    if ndim == 3:
        figsize = (10, 10)
    else:
        figsize = (10, 15)
    fig, rows = plt.subplots(ndim, len(METHODS), figsize=figsize, sharex=True)

    if isinstance(pdf, MixtureModel):
        mus = np.array([pdf.pdfs[i].mu for i in range(len(pdf.pdfs))])
    elif isinstance(pdf, VonMisesFisher):
        mus = pdf.mu

    # reference samples
    t = np.linspace(-1.0, 1.0, 1000)

    for d_idx, axes in enumerate(rows):

        # mixture of the marginals of von Mises-Fisher as ground truth samples
        marginalvMFs = [MarginalVonMisesFisher(d_idx, mu) for mu in mus]
        mixture_marginalvMFs = MixtureModel(marginalvMFs)
        log_p = mixture_marginalvMFs.log_prob(t)
        prob_truth = np.exp(log_p)

        # show histogram
        for ax, method in zip(axes, METHODS):
            marginals = samples[method][:, d_idx]
            bins = ax.hist(
                marginals,
                bins=bins,
                density=True,
                alpha=0.3,
                color="k",
                histtype="stepfilled",
            )[1]
            ax.plot(t, prob_truth, ls="--", c="r", lw=1)
            ax.set_xlabel(rf"$e_{d_idx}^Tx_n$", fontsize=fs)

    for ax, method in zip(rows[0], METHODS):
        ax.set_title(ALGOS[method], fontsize=fs)
    fig.tight_layout()

    if save_res:
        print("Saving histogram plot with mixture marginals..")
        fig.savefig(f"{path}/{filename}_hist_marginals.pdf", transparent=True)


def trace_plots(samples, ndim, path, filename, fs=16, save_res=False):
    """
    trace plots per dimension
    """

    os.makedirs(f"{path}/trace_plots", exist_ok=True)

    for d in range(ndim):
        fig, axes = plt.subplots(
            1, len(METHODS), figsize=(12, 5), sharex=True, sharey=True
        )

        for ax, method in zip(axes, METHODS):
            ax.set_title(ALGOS[method], fontsize=fs)
            ax.plot(samples[method][:, d], alpha=0.5, color="k", lw=1)
            ax.set_xlabel(r"MCMC step $n$", fontsize=fs)
        axes[0].set_ylabel(r"$u_{d}^Tx_n$", fontsize=fs)
        fig.suptitle(rf"Trace plot $d_{{{d+1}}}$")
        if save_res:
            print(f"saving trace plot for dimension {d+1}..")
            fig.savefig(
                f"{path}/trace_plots/{filename}_trace_d{d+1}.pdf", transparent=True
            )

        plt.close(fig)


def acf_plots(samples, ndim, path, filename, lag=1000, fs=16, save_res=False):
    """
    Plots acf for every dimension in a multidimensional target
    """

    os.makedirs(f"{path}/acf_plots", exist_ok=True)

    for d in range(ndim):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        for method in METHODS:
            ac = acf(samples[method][:, d], lag)
            ax.plot(ac, alpha=0.7, lw=3, label=ALGOS[method])
            ax.legend(fontsize=fs)
            ax.axhline(0.0, ls="--", color="k", alpha=0.7)
            ax.set_xlabel(r"Lag", fontsize=fs)
            ax.set_ylabel("ACF", fontsize=fs)
            ax.set_title(rf"$d_{{{d+1}}}$")
        if save_res:
            print(f"saving acf plots for dim {d+1}..")
            fig.savefig(f"{path}/acf_plots/{filename}_acf_d{d+1}.pdf", transparent=True)

        plt.close(fig)


def acf_entropy_plot(samples, pdf, path, filename, lag=1000, fs=16, save_res=False):
    """
    ACF entropy plot
    """

    # population of modes
    modes = np.array([p.mu for p in pdf.pdfs])

    # kl-divergence and entropy
    KL = []
    H = []
    for method in METHODS:
        x = samples[method]
        m = np.argmax(x @ modes.T, axis=1)
        i, c = np.unique(m, return_counts=True)
        p = np.full(len(modes), 1e-100)
        p[i] = c
        p[i] /= p.sum()
        KL.append(p @ np.log(p / pdf.weights))
        H.append(-(p @ np.log(p)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for method in METHODS:
        ac = acf(samples[method][:, 0], lag)
        ax.plot(ac, alpha=0.7, lw=3, label=ALGOS[method])
    ax.legend(fontsize=fs)
    ax.axhline(0.0, ls="--", color="k", alpha=0.7)
    ax.set_xlabel(r"Lag", fontsize=fs)
    ax.set_ylabel("ACF", fontsize=fs)

    # hopping frequency as bar plot
    ax = axes[1]
    ax.set_ylabel("Entropy")
    ax.bar(list(map(ALGOS.get, METHODS)), H, color="k", alpha=0.3)
    ax.axhline(np.log(len(pdf.pdfs)), ls="--", color="r", label="max entropy")
    plt.xticks(rotation=30)
    fig.tight_layout()

    if save_res:
        print("Saving ACF-Entropy plot..")
        fig.savefig(f"{path}/{filename}_acf_entropy.pdf", transparent=True)

    plt.close(fig)


def entropy_kld(samples, pdf, path, filename, save_res=False):
    """
    Plots entropy and KL divergence plot between
    target and sampled distribution
    """

    # population of modes
    modes = np.array([p.mu for p in pdf.pdfs])
    KL = []
    H = []
    for method in METHODS:
        x = samples[method]
        m = np.argmax(x @ modes.T, axis=1)
        i, c = np.unique(m, return_counts=True)
        p = np.full(len(modes), 1e-100)
        p[i] = c
        p[i] /= p.sum()
        KL.append(p @ np.log(p / pdf.weights))
        H.append(-(p @ np.log(p)))

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    ax = axes
    ax.set_ylabel("KL divergence")
    ax.bar(list(map(ALGOS.get, METHODS)), KL, color="k", alpha=0.3)
    plt.xticks(rotation=30)
    fig.tight_layout()
    if save_res:
        print("Saving KL divergence plot..")
        fig.savefig(f"{path}/{filename}_kl.pdf", bbox_inches="tight", transparent=True)

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    ax = axes
    ax.set_ylabel("Entropy")
    ax.bar(list(map(ALGOS.get, METHODS)), H, color="k", alpha=0.3)
    plt.xticks(rotation=30)
    fig.tight_layout()
    if save_res:
        print("Saving entropy plot..")
        fig.savefig(
            f"{path}/{filename}_entropy.pdf", bbox_inches="tight", transparent=True
        )

    plt.close(fig)


def dist_plot(samples, pdf, kappa, path, filename, fs=16, save_res=False):
    """
    plotting Geodesic distance
    """

    modes = np.array([p.mu for p in pdf.pdfs])
    d_modes = np.concatenate([distance(modes / kappa, mode / kappa) for mode in modes])
    d_modes = np.unique(np.round(d_modes, 5))
    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(len(METHODS) * 3, 3), sharex=True, sharey=True
    )
    bins = 100
    for ax, method in zip(axes, METHODS):
        ax.set_title(ALGOS[method], fontsize=fs)

        # distance between successive samples
        x = samples[method]
        d = distance(x[:-1], x[1:])
        print(
            "average great circle distance of successive samples: "
            f"{np.mean(d):.2f} ({method})"
        )

        # plot distance as histogram
        bins = ax.hist(
            d, bins=bins, density=True, alpha=0.3, color="k", histtype="stepfilled"
        )[1]
        for d_mode in d_modes[:0]:
            ax.axvline(d_mode, ls="--", color="r")
        ax.set_xlabel(r"$\delta(x_{n+1}, x_n)$", fontsize=fs)
        ax.set_xticks(np.linspace(0.0, np.pi, 3))
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        ax.semilogy()
    fig.tight_layout()

    if save_res:
        print("Saving geodesic distance plot..")
        fig.savefig(f"{path}/{filename}_dist.pdf", transparent=True)

    plt.close(fig)


def acf_kld_dist_plot(samples, pdf, path, filename, lag=80000, fs=16, save_res=False):
    """
    ACF entropy plot
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    # population of modes
    modes = np.array([p.mu for p in pdf.pdfs])

    # kl-divergence
    KL = []
    for method in METHODS:
        x = samples[method]
        m = np.argmax(x @ modes.T, axis=1)
        i, c = np.unique(m, return_counts=True)
        p = np.full(len(modes), 1e-100)
        p[i] = c
        p[i] /= p.sum()
        KL.append(p @ np.log(p / pdf.weights))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for method in METHODS:
        ac = acf(samples[method][:, 0], lag)
        ax.plot(ac, alpha=0.7, lw=3, label=ALGOS[method])
    ax.axhline(0.0, ls="--", color="k", alpha=0.7)
    ax.set_xlabel(r"Lag", fontsize=fs)
    ax.set_ylabel("ACF", fontsize=fs)
    ax.legend(fontsize=fs, loc="upper right")
    ax.tick_params(axis="both", which="major", labelsize=fs)

    # KL divergence
    ax2 = axes[1]
    ax2.set_ylabel("KL divergence")
    ax2.bar(list(map(ALGOS.get, METHODS)), KL, color="k", alpha=0.3)
    ax2.tick_params(axis="x", labelrotation=30)
    ax2.tick_params(axis="both", labelsize=fs)

    ax3 = axes[2]

    # Prepare the data for the histogram
    geo_dist_list = []
    for method in METHODS:
        x = samples[method]
        # Compute geodesic distances between successive samples
        geo_dist = distance(x[:-1], x[1:])
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
        alpha=0.3,
        ax=ax3,
        # palette=method_color_dict,
        legend=True,  # Ensure legend is enabled
    )

    # Customize the x-axis labels and ticks
    ax3.set_xlabel(r"$\delta(x_{n+1}, x_n)$", fontsize=20)
    ax3.set_xticks([0, np.pi / 2, np.pi])
    ax3.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"], fontsize=20)
    ax3.tick_params(axis="both", which="major", labelsize=fs)

    # Set y-scale to logarithmic
    ax3.set_yscale("log")
    ax3.set_ylabel(None)  # Remove the y-axis label
    ax3.set_xlim(0, np.pi)

    # Customize the legend
    leg = ax3.get_legend()
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

    if save_res:
        print("Saving ACF-KLD-geodesic distance plot..")
        fig.savefig(f"{path}/{filename}_acf_kld_dist.pdf", transparent=True)


def calc_ess(runs_samples, methods, path, return_ess=True):
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
                    # samples from all runs with shape (chains, draws, dimensions)
                    samples = np.array([draws[method] for draws in runs_samples])

                    # estimates ESS using the arviz library per dimension
                    samples_az = az.convert_to_dataset(samples)
                    ess_dims = az.ess(samples_az, relative=True)
                    ess[method] = ess_dims.x.values

                print(f"Saving ESS file {ess_file}")
                dump(ess, ess_file, gzip=True)

            else:
                print("ESS values not computed, requires `n_runs=10`")
                return None

    for method in methods:
        for i, vals in enumerate(ess[method]):
            print(f"{method} ESS dim {i+1}: {vals:.8%}")

    if return_ess:
        return ess
