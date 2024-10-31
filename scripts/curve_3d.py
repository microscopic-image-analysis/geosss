# samplers compared on the curve on a 2-sphere
import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from csb.io import dump, load
from scipy.spatial import cKDTree
from scipy.special import logsumexp

import geosss as gs
from geosss.distributions import CurvedVonMisesFisher, Distribution
from geosss.spherical_curve import SlerpCurve, constrained_brownian_curve

plt.rc("font", size=16)


def saff_sphere(N: int = 1000) -> np.ndarray:
    """Uniformly distribute points on the 2-sphere using Saff's algorithm."""
    h = np.linspace(-1, 1, N)
    theta = np.arccos(h)
    incr = 3.6 / np.sqrt(N * (1 - h[1:-1] ** 2))
    phi = np.add.accumulate(np.append(0, incr))
    phi = np.append(phi, 0.0)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return np.transpose([x, y, z])


def test_gradient(pdf):
    x = gs.sphere.sample_sphere()
    a = pdf.gradient(x)
    b = opt.approx_fprime(x, pdf.log_prob, 1e-7)
    print(a)
    print(b)
    assert np.allclose(a, b)

    # optional check (internally compares using `appox_fprime`)
    # error should be low!
    err = opt.check_grad(pdf.log_prob, pdf.gradient, x, seed=342)
    print(f"error to check correctness of gradient:, {err}")


def setup_logging(savedir: str, kappa: float, filemode: str = "a"):
    """Setting up logging

    Parameters
    ----------
    savedir : str
        log file directory
    kappa : float
        concentration parameter
    """
    logpath = f"{savedir}/curve_kappa{int(kappa)}_log.txt"
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


def _start_sampling(
    methods: str,
    tester: gs.SamplerLauncher,
    pdf: Distribution,
    savepath_samples: str,
    savepath_logprob: str,
):
    """just a util function to pass the output of this in a log file."""
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


def launch_samplers(
    savedir: str,
    kappa: float,
    pdf: Distribution,
    tester: gs.SamplerLauncher,
    methods: dict,
    rerun_if_file_exists: bool = False,
):
    """just an interface to load or run samplers"""

    savepath_samples = f"{savedir}/curve_samples_kappa{int(kappa)}.pkl"
    savepath_logprob = f"{savedir}/curve_logprob_kappa{int(kappa)}.pkl"

    # load existing files
    if (
        not rerun_if_file_exists
        and os.path.exists(savepath_samples)
        and os.path.exists(savepath_logprob)
    ):
        samples = load(savepath_samples)
        logging.info(f"Loading file {savepath_samples}")

        logprob = load(savepath_logprob)
        logging.info(f"Loading file {savepath_logprob}")

    # or run samplers
    else:
        samples, logprob = _start_sampling(
            methods,
            tester,
            pdf,
            savepath_samples,
            savepath_logprob,
        )

    return samples, logprob


def calc_kld(pdf, samples, methods, n_saff=1500):
    """estimating kl divergence for the curve vMF"""

    x = saff_sphere(n_saff)
    log_p = pdf.log_prob(x)
    p = np.exp(log_p - logsumexp(log_p))
    tree = cKDTree(x)

    kld = []
    for method in methods:
        d, i = tree.query(samples[method], k=1)
        j, c = np.unique(i, return_counts=True)
        q = np.zeros_like(p)
        q[j] = c = c / c.sum()
        kld.append(np.sum(p * np.log(p) - p * np.log(q + p.min())))
        print(method, kld[-1])

    return kld


def acf_kld_plot(
    pdf,
    samples,
    methods,
    algos,
    acf_lag=250,
    n_saff=1500,
    fs=16,
):
    """plots ACF for first dimension and KL divergence"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plotting acf for the first dimension
    for method in methods:
        ac = gs.acf(samples[method][:, 0], acf_lag)
        ax1.plot(ac, alpha=0.7, lw=3, label=algos[method])
    ax1.legend(fontsize=fs)
    ax1.axhline(0.0, ls="--", color="k", alpha=0.7)
    ax1.set_xlabel(r"Lag", fontsize=fs)
    ax1.set_ylabel("ACF", fontsize=fs)

    # calculate kl divergence
    kld = calc_kld(pdf, samples, methods, n_saff)

    # ax.set_title("KL divergence between target and sampled distribution")
    ax2.set_ylabel("KL divergence", fontsize=fs)
    ax2.bar(list(map(algos.get, methods)), kld, color="k", alpha=0.3)
    plt.xticks(rotation=30)
    fig.tight_layout()

    return fig


def geodesic_distance_plot(samples, methods, algos):
    """Geodesic distance between successive samples"""

    # geodesic distance
    fs = 16
    fig, axes = plt.subplots(
        1, len(methods), figsize=(len(methods) * 3, 3), sharex=True, sharey=True
    )
    bins = 100
    for ax, method in zip(axes, methods):
        ax.set_title(algos[method], fontsize=fs)
        # distance between successive samples
        x = samples[method]
        d = gs.distance(x[:-1], x[1:])
        logging.info(
            "average great circle distance of successive samples: "
            f"{np.mean(d):.2f} ({method})"
        )
        bins = ax.hist(
            d, bins=bins, density=True, alpha=0.3, color="k", histtype="stepfilled"
        )[1]
        ax.set_xlabel(r"$\delta(x_{n+1}, x_n)$", fontsize=fs)
        ax.set_xticks(np.linspace(0.0, np.pi, 3))
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        ax.semilogy()
    fig.tight_layout()

    return fig


def acf_geodist_kld_plot(
    samples,
    methods,
    algos,
    savedir,
    filename="curve_acf_distplot",
    savefig=True,
    acf_lag=1000,
    n_saff=1500,
):

    # Suppress FutureWarnings (optional)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Define your methods and corresponding colors
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    method_color_dict = dict(zip(methods, colors))

    # Font size for labels and titles
    fs = 16

    # Create the figure with two subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # first subplot : ACF
    ax1 = axes[0]

    for method, color in zip(methods, colors):
        ac = gs.acf(samples[method][:, 0], acf_lag)
        ax1.plot(ac, alpha=0.7, lw=3, label=algos[method], color=color)

    ax1.axhline(0.0, ls="--", color="k", alpha=0.7)
    ax1.set_xlabel(r"Lag", fontsize=fs)
    ax1.set_ylabel("ACF", fontsize=fs)
    ax1.tick_params(axis="both", which="major", labelsize=fs)
    ax1.legend(fontsize=fs, loc="upper right")

    # Second Subplot: KL divergence
    ax2 = axes[1]

    # calculate kl divergence
    kld = calc_kld(pdf, samples, methods, n_saff)

    # ax.set_title("KL divergence between target and sampled distribution")
    ax2.set_ylabel("KL divergence", fontsize=fs)
    ax2.bar(list(map(algos.get, methods)), kld, color="k", alpha=0.3)
    ax2.tick_params(axis="x", labelrotation=30)
    ax2.tick_params(axis="both", labelsize=fs)
    fig.tight_layout()

    # Third Subplot: Geodesic distance
    ax3 = axes[2]

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

    if savefig:
        logging.info(
            f"Saving ACF, KL divergence and geodesic distance plot to {savedir}/{filename}.pdf"
        )
        fig.savefig(
            f"{savedir}/{filename}.pdf",
            bbox_inches="tight",
            transparent=True,
            dpi=150,
        )


def marginal_distribution_plot(pdf, samples, methods, algos):
    """marginal distribution plot"""

    x = saff_sphere(100_000)
    log_p = pdf.log_prob(x)
    prob_truth = np.exp(log_p - logsumexp(log_p))

    bins = 50
    fs = 16
    plt.rc("font", size=fs)
    fig, rows = plt.subplots(
        3, len(methods), figsize=(12, 10), sharex=True, sharey=True
    )
    for i, axes in enumerate(rows):
        vals = x[:, i]
        ref = list(np.histogram(vals, weights=prob_truth, bins=bins, density=True))
        ref[1] = 0.5 * (ref[1][1:] + ref[1][:-1])
        for ax, method in zip(axes, methods):
            bins = ax.hist(
                samples[method][burnin:, i],
                bins=bins,
                density=True,
                alpha=0.3,
                color="k",
                histtype="stepfilled",
            )[1]
            ax.plot(*ref[::-1], color="r", lw=1, ls="--")
            ax.set_xlabel(rf"$e_{i}^Tx_n$", fontsize=fs)
    for ax, method in zip(rows[0], methods):
        ax.set_title(algos[method], fontsize=fs)
    fig.tight_layout()

    return fig


def scatter_curve_3d(
    pdf: Distribution,
    samples: dict,
    METHODS: tuple,
    ALGOS: dict,
    n_saff_samples: int = 30000,
    fontsize=16,
    elev=74,
    azim=-4,
):
    """Visualizing the density and the samples on a 2-sphere

    Parameters
    ----------
    pdf : Distribution
        unnormalized probability density function
    samples : dict
        samples corresponding to each method
    METHODS : tuple
        MCMC methods used here
    ALGOS : dict
        name of the MCMC methods (used for titles)
    n_saff_samples : int, optional
        number of samples for uniformly sampling the sphere, by default 30000
    fontsize : int, optional
        default font size, by default 16
    elev : int, optional
        polar angle for viewing the 3D plot, by default 62
    azim : int, optional
        azimuthal angle for viewing the 3D plot, by default 11

    Returns
    -------
    Figure
        returns the 3D plot
    """
    for method in METHODS:
        assert samples[method].shape[1] == 3, "Visualization accepts only 3D samples."

    # true PDF values
    saff_samples = saff_sphere(n_saff_samples)
    log_p = pdf.log_prob(saff_samples)
    prob_truth = np.exp(log_p - logsumexp(log_p))

    # Create a regular grid over theta and phi for the wire-plot
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(0, np.pi, 300), np.linspace(0, 2 * np.pi, 300)
    )

    # Convert the spherical grid to Cartesian coordinates
    X = np.sin(theta_grid) * np.cos(phi_grid)
    Y = np.sin(theta_grid) * np.sin(phi_grid)
    Z = np.cos(theta_grid)

    # Normalize for color mapping
    fig, axes = plt.subplots(
        1,
        len(METHODS),
        figsize=(16, 6),
        subplot_kw={"projection": "3d"},
        sharex=True,
        sharey=True,
    )

    # Compute the viewing vector
    def _get_view_vector(elev, azim):
        """Convert elevation and azimuth to viewing vector."""
        elev_rad = np.deg2rad(elev)
        azim_rad = np.deg2rad(azim)

        # Adjust azimuth to match Matplotlib's convention
        x = np.cos(elev_rad) * np.cos(azim_rad)
        y = np.cos(elev_rad) * np.sin(azim_rad)
        z = np.sin(elev_rad)
        return np.array([x, y, z])

    # View vector
    view_vector = _get_view_vector(elev, azim)

    for ax, method in zip(axes.flat, METHODS):
        ax.computed_zorder = False
        ax.scatter(
            *saff_samples.T, c=prob_truth, s=10, alpha=1.0, cmap="terrain_r", zorder=1
        )
        # Select the first 10000 samples
        sample_points = samples[method][:10000]
        dot_products = np.dot(sample_points, view_vector)

        # Map dot products to alpha values
        # Desired alpha range: 0.0 (fully transparent) to max_alpha (for visible points)
        min_alpha = 0.0  # Minimum alpha for back-facing points
        max_alpha = 0.16  # Maximum alpha for front-facing points

        # Normalize dot products from [-1, 1] to [min_alpha, max_alpha]
        alpha_values = min_alpha + ((dot_products + 1) / 2) * (max_alpha - min_alpha)
        alpha_values = np.clip(alpha_values, min_alpha, max_alpha)

        # Create colors with varying alpha, base color black
        colors = np.zeros((sample_points.shape[0], 4))
        colors[:, :3] = 0  # for black color
        colors[:, 3] = alpha_values  # custom alpha values

        # scatter points and specify the custom colors
        ax.scatter(*sample_points.T, c=colors, s=1, zorder=2)
        ax.plot_wireframe(X, Y, Z, color="lightgrey", alpha=0.05, zorder=3)

        ax.set_title(ALGOS[method], pad=-50, fontsize=fontsize)
        ax.set_aspect("auto")
        ax.view_init(elev, azim)
        ax.axis("off")

    plt.subplots_adjust(wspace=-0.1, hspace=-0.2)

    return fig


if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process parameters for the curve generation."
    )

    # Add arguments for kappa and n_samples
    parser.add_argument(
        "--kappa",
        type=float,
        default=300.0,
        help="Concentration parameter (default: 300.0)",
    )
    parser.add_argument(
        "--n_samples",
        type=float,
        default=1e5,
        help="Number of samples per sampler (default: 1000)",
    )

    # Parse arguments
    args = parser.parse_args()

    # parameters
    kappa = args.kappa  # concentration parameter
    n_samples = int(args.n_samples)  # number of samples per sampler
    burnin = int(0.1 * n_samples)  # burn-in

    # optional controls
    is_brownian_curve = False  # fix curve (target)
    reprod_switch = True  # seeds samplers for reproducibility
    savefig = True  # save the plots
    rerun_if_file_exists = False  # rerun even if file exists

    # directory to save results and log info
    savedir = f"results/vMF_curve_3d_kappa{int(kappa)}"
    os.makedirs(savedir, exist_ok=True)
    setup_logging(savedir, kappa)

    # define curve on the sphere
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
    else:
        knots = constrained_brownian_curve(
            n_points=10,
            dimension=3,
            step_size=0.3,
            seed=72367 if reprod_switch else None,
        )

    curve = SlerpCurve(knots)
    pdf = CurvedVonMisesFisher(curve, kappa)

    # eval density
    x = saff_sphere(5000)
    log_p = pdf.log_prob(x)
    prob_truth = np.exp(log_p - logsumexp(log_p))
    t = np.linspace(0, 1, 1_000)  # points on curve

    # show curve on the sphere
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="3d"))
    ax.set_box_aspect((1, 1, 1))
    ax.scatter(*x.T, c=prob_truth, s=10, alpha=0.15)
    ax.plot(*curve(t).T, color="k", alpha=1.0)
    ax.scatter(*curve(t).T, c=t, s=1)
    ax.scatter(*curve.knots.T, c="r", s=20)
    fig.tight_layout()

    # initial state fixed and samplers seeded for reproducibility
    initial = np.array([0.65656515, -0.63315859, -0.40991755])
    seed = 6756 if reprod_switch else None

    # `tester` instances samplers
    launcher = gs.SamplerLauncher(pdf, initial, n_samples, burnin, seed)
    methods = ("sss-reject", "sss-shrink", "rwmh", "hmc")
    algos = {
        "sss-reject": "geoSSS (reject)",
        "sss-shrink": "geoSSS (shrink)",
        "rwmh": "RWMH",
        "hmc": "HMC",
    }

    # load samples by running or loading from memory
    samples, logprob = launch_samplers(
        savedir,
        kappa,
        pdf,
        launcher,
        methods,
        rerun_if_file_exists,
    )

    fig = scatter_curve_3d(pdf, samples, methods, algos)
    if savefig:
        fig.savefig(
            f"{savedir}/scattered_3d_curve_kappa{int(kappa)}.png",
            bbox_inches="tight",
            transparent=True,
            dpi=300,
        )

    fig2 = acf_kld_plot(pdf, samples, methods, algos, acf_lag=250)
    if savefig:
        fig2.savefig(
            f"{savedir}/curve_acf_kld_kappa{int(kappa)}.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    fig3 = geodesic_distance_plot(samples, methods, algos)
    if savefig:
        fig3.savefig(
            f"{savedir}/curve_dist_kappa{int(kappa)}.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    fig4 = marginal_distribution_plot(pdf, samples, methods, algos)
    if savefig:
        fig4.savefig(
            f"{savedir}/curve_hist_kappa{int(kappa)}.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    fig5 = acf_geodist_kld_plot(
        samples,
        methods,
        algos,
        savedir,
        f"curve_acf_kld_geodist_3d_kappa{int(kappa)}",
        savefig=savefig,
        acf_lag=250,
    )

    # some misc plots (either redundant or not used in paper)
    misc_plots = False

    if misc_plots:
        # generate figures
        fs = 16
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for d, ax in enumerate(axes):
            ax.set_title(rf"$x_{d+1}$", fontsize=20)
            for method in methods:
                ac = gs.acf(samples[method][:, d], 250)
                ax.plot(ac, alpha=0.7, lw=3, label=algos[method])
            ax.axhline(0.0, ls="--", color="k", alpha=0.7)
            ax.set_xlabel(r"Lag", fontsize=fs)
        axes[0].set_ylabel("ACF", fontsize=fs)
        ax.legend(fontsize=fs)
        fig.tight_layout()

        # compare histograms

        x = saff_sphere(1500)
        log_p = pdf.log_prob(x)
        prob_truth = np.exp(log_p - logsumexp(log_p))

        tree = cKDTree(x)

        kl_methods = []
        for method in methods:
            d, i = tree.query(samples[method], k=1)
            j, c = np.unique(i, return_counts=True)
            q = np.zeros_like(prob_truth)
            q[j] = c = c / c.sum()
            kl = np.sum(
                prob_truth * np.log(prob_truth)
                - prob_truth * np.log(q + prob_truth.min())
            )
            kl_methods.append(kl)
            logging.info(f"KL divergence between target and {method}: {kl_methods[-1]}")

        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        ax = axes
        # ax.set_title("KL divergence between target and sampled distribution")
        ax.set_ylabel("KL divergence")
        ax.bar(list(map(algos.get, methods)), kl_methods, color="k", alpha=0.3)
        plt.xticks(rotation=30)
        fig.tight_layout()
