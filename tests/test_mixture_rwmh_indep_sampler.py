# For simple testing of the implementation of the "mixture kernel sampler" on
# "mixture of von Mises-Fisher distributions"

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arviz import autocorr
from matplotlib.colors import Normalize

import geosss as gs
from geosss.distributions import Distribution
from geosss.sphere import distance

METHODS = ("sss-reject", "sss-shrink", "rwmh", "hmc", "mix-rwmh-indep")
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
    "mix-rwmh-indep": "mixture RWMH - \nIndependence sampler",
}


def sphere_pdf(n_grid: int, pdf: Distribution):
    """spherical grid with pdf"""

    # spherical grid
    u = np.linspace(0, np.pi, n_grid)
    v = np.linspace(0, 2 * np.pi, n_grid)

    u_grid, v_grid = np.meshgrid(u, v)
    vertices = np.stack(
        [
            np.cos(v_grid) * np.sin(u_grid),
            np.sin(v_grid) * np.sin(u_grid),
            np.cos(u_grid),
        ],
        axis=2,
    )

    # spherical to cartesian
    sph2cart = (
        np.outer(np.cos(v), np.sin(u)),
        np.outer(np.sin(v), np.sin(u)),
        np.outer(np.ones_like(u), np.cos(u)),
    )

    # pdf values for the grid
    pdf_vals = np.array([np.exp(pdf.log_prob(val)) for val in vertices])

    return sph2cart, pdf_vals


def compare_samplers_3d(
    pdf: Distribution,
    samples: dict,
    n_grid: int = 100,
):
    for method in METHODS:
        msg = "visualization accepts only 3 dimension"
        assert samples[method].shape[1] == 3, msg

    sph2cart, pdf_vals = sphere_pdf(n_grid, pdf)
    pdfnorm = Normalize(vmin=pdf_vals.min(), vmax=pdf_vals.max())

    fig, axes = plt.subplots(
        1, 5, figsize=(12, 4), subplot_kw={"projection": "3d"}, sharex=True, sharey=True
    )

    for ax, method in zip(axes.flat, METHODS):
        ax.computed_zorder = False
        ax.plot_surface(
            *sph2cart, facecolors=plt.cm.terrain_r(pdfnorm(pdf_vals)), alpha=1, zorder=1
        )

        x = samples[method]
        ax.scatter(*x.T, c="tab:red", s=1, alpha=0.4, zorder=2)
        ax.set_title(ALGOS[method], pad=-30)
        ax.set_aspect("equal")
        ax.view_init(-140, 150)
        ax.axis("off")

    fig.tight_layout()


def acf_kld_dist_plot(
    samples: dict,
    pdf: gs.VonMisesFisher,
    lag: int = 80000,
    fs: int = 16,
):
    """
    ACF-KLD and geodesic distance plot
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    method_color_dict = dict(zip(METHODS, colors))

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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    for method, color in zip(METHODS, colors):
        ac = autocorr(samples[method][:, 0])[:lag]
        ax.plot(
            ac,
            alpha=0.7,
            lw=3,
            label=ALGOS[method],
            color=color,
        )
    ax.axhline(0.0, ls="--", color="k", alpha=0.7)
    ax.set_xlabel(r"Lag", fontsize=fs)
    ax.set_ylabel("ACF", fontsize=fs)
    ax.legend(fontsize=fs, loc="upper right")
    ax.tick_params(axis="both", which="major", labelsize=fs)

    # KL divergence
    ax2 = axes[1]
    ax2.set_ylabel("KL divergence")
    ax2.bar(
        list(map(ALGOS.get, METHODS)),
        KL,
        color=colors,
        alpha=0.5,
        edgecolor=colors,
        linewidth=2,
    )
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
        palette=method_color_dict,
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


if __name__ == "__main__":
    # parameters for mixture of von Mises-Fisher (vMF)
    # distributions
    d = 3  # required dimension
    K = 3  # number of mixture components
    kappa = 80.0  # concentration parameter

    # mus (mean directions) of the vMF mixture components for a 2-sphere
    # represented as a unit quaternion.
    mus = np.array(
        [
            [0.86981638, -0.37077248, 0.32549536],
            [-0.19772391, -0.89279985, -0.40473902],
            [0.19047726, 0.22240888, -0.95616562],
        ]
    )

    # sampler parameters
    n_samples = int(1e3)  # no. of samples
    burnin = int(0.1 * n_samples)  # burnin samples
    seed = 3521  # sampler seed

    # target pdf
    vmfs = [gs.VonMisesFisher(kappa * mu) for mu in mus]
    pdf = gs.MixtureModel(vmfs)

    # initial state of the samplers
    init_state = np.array([-0.86333052, 0.18685286, -0.46877117])

    # sampling with the four samplers
    samples = {}

    # Sample with different methods (RWMH, HMC are automatically tuned in step-size)
    samplers = {
        "sss-reject": gs.RejectionSphericalSliceSampler,  # very accurate, but slow
        "sss-shrink": gs.ShrinkageSphericalSliceSampler,  # reasonably accurate, but fast
        "rwmh": gs.MetropolisHastings,
        "hmc": gs.SphericalHMC,
    }

    mixing_probabilities = np.arange(0.1, 1.1, 0.1)

    for alpha in mixing_probabilities:
        print(f"Mix sampler with alpha: {alpha}")
        sampler_mix = gs.MixtureRWMHIndependenceSampler(
            pdf,
            init_state,
            seed,
            mixing_probability=alpha,
        )

        samples = {
            name: cls(pdf, init_state, seed).sample(n_samples, burnin)
            for name, cls in samplers.items()
        }

        samples["mix-rwmh-indep"] = sampler_mix.sample(n_samples, burnin)

        # Verify burnin adaptation
        print("\n=== Verifications for mix-rwmh-indep ===")
        print(f"Mix sampler with alpha: {alpha}")
        print(f"\nTotal samples: {n_samples + burnin}")
        print(f"Step size adapted over (burnin): {sampler_mix._counter}")
        print(f"Final stepsize: {sampler_mix.stepsize:.6f}")
        print(f"Total RWMH proposals: {sampler_mix.rwmh_counter}")
        print(f"Total independence proposals: {sampler_mix.indep_counter}")
        print(f"Total acceptances: {sampler_mix.n_accept}")
        print(
            f"Overall acceptance rate: {sampler_mix.n_accept / (n_samples + burnin):.2%}"
        )

        compare_samplers_3d(pdf, samples)

        acf_kld_dist_plot(samples, pdf, lag=None)
        plt.show()

# %%
