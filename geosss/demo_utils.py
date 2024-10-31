import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from geosss.distributions import Distribution

METHODS = ("sss-reject", "sss-shrink", "rwmh", "hmc")
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
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
        2, 2, figsize=(8, 8), subplot_kw={"projection": "3d"}, sharex=True, sharey=True
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
    plt.show()
