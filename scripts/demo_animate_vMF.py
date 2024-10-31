import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

import geosss as gs

methods = ("sss-reject", "sss-shrink", "rwmh", "hmc")
algos = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
}


def show_sphere(pdf: gs.Distribution, n_grid: int = 1000):
    """
    Util function for `animate_vMF` that shows four spheres as surface
    plots with a colormap based on the pdf.

    Args:
        pdf (gs.Distribution): probability density function
        n_grid (int, optional): no. of grid on the sphere. Defaults to 1000.

    """
    sph2cart, pdf_vals = gs.sphere_pdf(n_grid, pdf)
    pdfnorm = Normalize(vmin=pdf_vals.min(), vmax=pdf_vals.max())

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(16, 6),
        subplot_kw={"projection": "3d"},
        sharex=True,
        sharey=True,
    )

    for ax, method in zip(axes.flat, methods):
        ax.computed_zorder = False
        ax.plot_surface(
            *sph2cart,
            facecolors=plt.cm.terrain_r(pdfnorm(pdf_vals)),
            rstride=1,
            cstride=1,
            alpha=1.0,
            zorder=1,
            edgecolor="none",
            antialiased=True,
        )
        ax.set_title(algos[method], fontsize=15, pad=-50)
        ax.set_aspect("auto")
        ax.view_init(-140, 150)
        ax.axis("off")

    # Adjust layout to reduce gaps
    plt.subplots_adjust(wspace=-0.1, hspace=-0.2)

    return fig, axes


def animate_vMF(
    pdf: gs.Distribution,
    samples: dict,
    frames: int = 20,
    fps: int = 20,
    n_grid: int = 300,
    save_anim: bool = False,
    savepath: str = "results/animation_vMF",
):
    """
    Generates animation for samples on a 2-sphere using the
    four sampling methods in the package defined

    Args:
        pdf (gs.Distribution): probability density function
        samples (dict): samples saved as dict from the four samplers
        frames (int, optional): number of frames. Defaults to 20.
        fps (int, optional): frames per second. Defaults to 20.
        save_anim (bool, optional): whether to save animation. Defaults to False.
        savepath (str, optional): directory to save the animation. Defaults to "results/animation_vMF".

    """

    # determine which samples to show
    n_samples = np.max([len(samples[method]) for method in methods])
    frame_range = np.linspace(1, n_samples, frames, dtype=int)

    fig, axes = show_sphere(pdf, n_grid)

    def update(idx):
        # sample index updates faster
        samples_idx = frame_range[idx]

        for ax, method in zip(axes.flat, methods):
            x = samples[method][:samples_idx]
            ax.scatter(*x.T, c="tab:red", s=1, alpha=0.2, zorder=2)

    anim = FuncAnimation(fig, update, frames=frames, interval=1)

    if save_anim:
        anim.save(f"{savepath}", fps=fps, writer="pillow")

    return anim


if __name__ == "__main__":
    # parameters for mixture of von Mises-Fisher
    # distributions
    d = 3  # dimension
    K = 3  # number of mixture components
    kappa = 80.0  # concentration parameter
    run_anim = True  # switch to generate animation in README

    # mus of mixture components
    mus = np.array(
        [
            [0.86981638, -0.37077248, 0.32549536],
            [-0.19772391, -0.89279985, -0.40473902],
            [0.19047726, 0.22240888, -0.95616562],
        ]
    )

    # target pdf
    vmfs = [gs.VonMisesFisher(kappa * mu) for mu in mus]
    pdf = gs.MixtureModel(vmfs)

    # sampler parameters
    n_samples = int(1e3)  # no. of samples
    burnin = int(0.1 * n_samples)  # burnin samples
    sampler_seed = 3521  # sampler seed

    # interface for launching the four samplers
    # GeoSSS (reject), GeoSSS (shrink), RWMH and HMC
    init_state = np.array([-0.86333052, 0.18685286, -0.46877117])
    launcher = gs.SamplerLauncher(pdf, init_state, n_samples, burnin, sampler_seed)

    # save samples as dictionary
    samples = {}
    for method in methods:
        samples[method] = launcher.run(method)

    if run_anim:
        # animation params
        save_anim = True  # whether to save animation
        frames = 10  # no. of frames (set it relatively low)
        fps = 10  # frames per second

        # directory to save the result
        savepath = "results/animation_vMF.gif"

        # run animation
        anim = animate_vMF(
            pdf, samples, fps=fps, n_grid=100, save_anim=save_anim, savepath=savepath
        )

    # visualize samples in 3d
    gs.compare_samplers_3d(pdf, samples)
