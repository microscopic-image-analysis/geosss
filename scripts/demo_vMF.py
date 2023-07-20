import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

import geosss as gs

methods = ('sss-reject', 'sss-shrink', 'rwmh', 'hmc')
algos = {'sss-reject': 'geoSSS (reject)', 'sss-shrink': 'geoSSS (shrink)',
         'rwmh': 'RWMH', 'hmc': 'HMC'}


def show_sphere(pdf: gs.Distribution, n_grid: int = 100):

    sph2cart, pdf_vals = gs.sphere_pdf(n_grid, pdf)
    pdfnorm = Normalize(vmin=pdf_vals.min(), vmax=pdf_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={
        "projection": "3d"}, sharex=True, sharey=True)

    for ax, method in zip(axes.flat, methods):
        ax.computed_zorder = False
        ax.plot_surface(*sph2cart,
                        facecolors=plt.cm.terrain_r(pdfnorm(pdf_vals)),
                        alpha=0.8, zorder=1)
        ax.set_title(algos[method], fontsize=15)
        ax.set_aspect('equal')
        ax.view_init(-140, 150)
        ax.axis('off')

    fig.tight_layout()

    return fig, axes


def animate_vMF(
        pdf: gs.Distribution,
        samples: dict,
        fps: int = 20,
        save_animation: bool = False,
        filename: str = "animation_vMF"
):

    fig, axes = show_sphere(pdf, 100)

    def update(frame):
        for ax, method in zip(axes.flat, methods):
            x = samples[method][:frame*20]
            ax.scatter(*x.T, c='tab:red', s=1, alpha=0.2, zorder=2)

    anim = FuncAnimation(fig, update, frames=50, interval=1)

    if save_animation:
        anim.save(f"results/{filename}.gif",
                  fps=fps, dpi=300, writer="pillow")


if __name__ == '__main__':

    # parameters for mixture of von Mises-Fisher
    # distributions
    d = 3                          # dimension
    K = 3                          # number of mixture components
    kappa = 80.0                   # concentration parameter
    run_anim = False               # animation from README (Warning: very slow)

    # mus of mixture components
    mus = np.array([[0.86981638, -0.37077248,  0.32549536],
                    [-0.19772391, -0.89279985, -0.40473902],
                    [0.19047726,  0.22240888, -0.95616562]])

    # target pdf
    vmfs = [gs.VonMisesFisher(kappa*mu) for mu in mus]
    pdf = gs.MixtureModel(vmfs)

    # sampler parameters
    n_samples = int(1e3)           # no. of samples
    burnin = int(0.1 * n_samples)  # burnin samples
    sampler_seed = 3521            # sampler seed

    # interface for launching the four samplers
    # GeoSSS (reject), GeoSSS (shrink), RWMH and HMC
    init_state = np.array([-0.86333052,  0.18685286, -0.46877117])
    launcher = gs.SamplerLauncher(
        pdf,
        init_state,
        n_samples,
        burnin,
        sampler_seed
    )

    # save samples as dictionary
    samples = {}
    for method in methods:
        samples[method] = launcher.run(method)

    # visualize samples in 3d
    gs.compare_samplers_3d(pdf, samples)

    # sampling animation demo as gif (warning: slow)
    if run_anim:
        # animation params
        save_anim = False           # whether to save animation
        filename = "animation_vMF"  # filename of the gif file
        fps = 10                    # frames per second

        # run animation
        animate_vMF(pdf, samples, fps, save_anim, filename)

    plt.show()
