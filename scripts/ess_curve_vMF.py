# ESS computed for the curve on the sphere by varying the number of dimensions and
# the concentration parameter kappa.
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from geosss.io import dump, load

METHODS = ("sss-reject", "sss-shrink", "rwmh", "hmc")
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
}
plt.rc("font", size=11)


def load_samples(
    base_path,
    varying_param_values,
    varying_param_name,
    fixed_params,
    n_runs=10,
    verbose=False,
):
    """
    Load chains of samples for varying a parameter.

    Parameters
    ----------
    base_path : str
        Base directory where the samples are stored.
    varying_param_values : list
        List of values for the varying parameter (kappas or dimensions).
    varying_param_name : str
        Name of the varying parameter ('kappa' or 'n_dim').
    fixed_params : dict
        Dictionary of fixed parameters (e.g., {'n_dim': 10} or {'kappa': 500}).
    n_runs : int, optional
        Number of runs (chains), by default 10.
    verbose : bool, optional
        Whether to print loading progress, by default False.

    Returns
    -------
    dict
        Dictionary that can be read as data_dict[param_value][method]
    """
    # Initialize a dictionary to hold data for each varying parameter value and method
    data_dict = {
        param_value: {method: [] for method in METHODS}
        for param_value in varying_param_values
    }

    for param_value in varying_param_values:
        # Update the parameters with the varying parameter
        params = fixed_params.copy()
        params[varying_param_name] = param_value

        # Build subdir name based on parameters
        n_dim = params.get("n_dim")
        kappa = params.get("kappa")
        subdir = f"curve_{n_dim}d_kappa_{float(kappa)}"

        for chain in range(n_runs):
            # Build samples filename
            filename = f"curve_samples_{n_dim}d_kappa_{float(kappa)}_run{chain}.pkl"
            samples_file = os.path.join(base_path, subdir, filename)
            if os.path.exists(samples_file):
                samples_all = load(samples_file)
                print(f"Loading file {samples_file}") if verbose else None

                # Add to the main dictionary
                for method in METHODS:
                    if method in samples_all:
                        data_dict[param_value][method].append(samples_all[method])
            else:
                error_msg = (
                    f"Make sure samples are precomputed and stored for {samples_file}"
                )
                raise FileNotFoundError(error_msg)

    # Stack the lists into arrays
    for param_value in data_dict:
        for method in data_dict[param_value]:
            data_dict[param_value][method] = np.stack(
                data_dict[param_value][method], axis=0
            )

    return data_dict


def calc_ess(samples_dict, verbose=False):
    """
    Calculate the Effective Sample Size (ESS) for every dimension and for each method
    using the "bulk" method from the arviz package.

    Parameters
    ----------
    samples_dict : dict
        Dictionary containing samples for each method with shape (chains, draws, dimensions).
    verbose : bool

    Returns
    -------
    dict
        ESS values per dimension per method.
    """
    # Convert the dict to an ArviZ dataset
    samples_az = az.dict_to_dataset(samples_dict)

    # Ensure there are enough chains
    for method in METHODS:
        chains = samples_az[method].values.shape[0]
        assert chains >= 10, "ESS calculation requires at least 10 chains."

    ess_vals = {}
    for method in METHODS:
        samples = samples_az[method]
        ess = az.ess(samples, relative=True)[method]
        ess_vals[method] = ess
        if verbose:
            for i, val in enumerate(ess.values):
                print(f"{method} ESS dim {i + 1}: {val:.4%}")

    return ess_vals


def calc_ess_varying_param(
    param_values,
    datasets,
    ess_filepath,
    verbose=False,
):
    """
    Calculate or load ESS values for varying a parameter (kappas or dimensions).

    Parameters
    ----------
    param_values : list
        List of varying parameter values.
    datasets : dict
        Loaded datasets, indexed by parameter value.
    ess_filepath : str
        File path to save or load ESS values.
    verbose : bool

    Returns
    -------
    dict
        ESS values indexed by parameter value and method.
    """
    ess_vals = {}
    for param_value in param_values:
        print(f"Calculating ESS for {param_value=}")
        ess_vals[param_value] = calc_ess(datasets[param_value], verbose=verbose)
    dump(ess_vals, ess_filepath)

    return ess_vals


def ess_plot_varying_param(
    ess_vals,
    param_values,
    param_name,
    select_dim_idx: int = 0,
    y_lim_factor: float = 18.0,
) -> plt.Figure:
    """
    Plot ESS values against a varying parameter.

    Parameters
    ----------
    ess_vals : dict
        ESS values indexed by parameter value and method.
    param_values : list
        List of parameter values.
    param_name : str
        Name of the parameter for labeling.
    select_dim_idx : int
        Index to select the ESS values that are computed for every dimension.
    y_lim_factor : float
        Factor to multiply the y limit by.
    Returns
    -------
    plt.Figure
    """
    ess_single_dim = {method: [] for method in METHODS}
    for method in METHODS:
        for param_value in param_values:
            ess_val = ess_vals[param_value][method][select_dim_idx].values
            ess_single_dim[method].append(float(ess_val))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["8", "s", "^", "P"]
    color_palette = sns.color_palette("deep", n_colors=len(METHODS))
    for i, method in enumerate(METHODS):
        label = ALGOS[method]
        ax.plot(
            param_values,
            ess_single_dim[method],
            marker=markers[i],
            markersize=10,
            label=label,
            color=color_palette[i],
        )
    ax.set_yscale("log")

    # Adjust y limit
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * y_lim_factor)
    ax.legend(loc="upper right")

    ax.set_xlabel(param_name)
    ax.set_ylabel("relative ESS (log)")

    ax.set_xticks(param_values)
    ax.set_xticklabels(param_values)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Varying kappa (left subplot)
    kappas = np.arange(100, 900, 100)
    n_dim = 5
    n_runs = 10
    subdir_kappa = f"results/curve_{n_dim}d_vary_kappa_nruns_{n_runs}"
    ess_filename_kappa = f"ess_curve_{n_dim}d_varying_kappa.pkl"
    ess_filepath_kappa = os.path.join(subdir_kappa, ess_filename_kappa)
    recompute_ess = False

    # Load or calculate ESS for varying kappa
    if not recompute_ess and os.path.exists(ess_filepath_kappa):
        print("Loading ESS values for varying kappa from file...")
        ess_kappas = load(ess_filepath_kappa)
    else:
        print(f"Loading samples for varying kappa from {subdir_kappa}...")
        datasets_varying_kappa = load_samples(
            base_path=subdir_kappa,
            varying_param_values=kappas,
            varying_param_name="kappa",
            fixed_params={"n_dim": n_dim},
            n_runs=n_runs,
            verbose=True,
        )
        ess_kappas = calc_ess_varying_param(
            param_values=kappas,
            datasets=datasets_varying_kappa,
            ess_filepath=ess_filepath_kappa,
            verbose=True,
        )

    # Plot varying kappa on left subplot
    ess_single_dim_kappa = {method: [] for method in METHODS}
    for method in METHODS:
        for kappa_val in kappas:
            ess_val = ess_kappas[kappa_val][method][0].values  # select_dim_idx=0
            ess_single_dim_kappa[method].append(float(ess_val))

    markers = ["8", "s", "^", "P"]
    color_palette = sns.color_palette("deep", n_colors=len(METHODS))
    for i, method in enumerate(METHODS):
        label = ALGOS[method]
        ax1.plot(
            kappas,
            ess_single_dim_kappa[method],
            marker=markers[i],
            markersize=6,
            label=label,
            color=color_palette[i],
        )

    ax1.set_yscale("log")
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax * 28)  # y_lim_factor for kappa plot
    ax1.legend(loc="upper right")
    ax1.set_xlabel(r"concentration parameter $\kappa$")
    ax1.set_ylabel("relative ESS (log)")
    ax1.set_xticks(kappas)
    ax1.set_xticklabels(kappas)

    # Plot 2: Varying ndim (right subplot)
    kappa = 500.0
    ndims = np.arange(3, 27, 3)
    n_runs = 10
    subdir_ndim = f"results/curve_kappa_{float(kappa)}_vary_ndim_nruns_{n_runs}"
    ess_filename_ndim = f"ess_curve_kappa_{int(kappa)}_varying_ndim.pkl"
    ess_filepath_ndim = os.path.join(subdir_ndim, ess_filename_ndim)

    # Load or calculate ESS for varying ndim
    if not recompute_ess and os.path.exists(ess_filepath_ndim):
        print("Loading ESS values for varying ndim from file...")
        ess_ndims = load(ess_filepath_ndim)
    else:
        print(f"Loading samples for varying n_dim from {subdir_ndim}...")
        datasets_varying_ndim = load_samples(
            base_path=subdir_ndim,
            varying_param_values=ndims,
            varying_param_name="n_dim",
            fixed_params={"kappa": kappa},
            n_runs=n_runs,
            verbose=True,
        )
        ess_ndims = calc_ess_varying_param(
            param_values=ndims,
            datasets=datasets_varying_ndim,
            ess_filepath=ess_filepath_ndim,
            verbose=True,
        )

    # Plot varying ndim on right subplot
    ess_single_dim_ndim = {method: [] for method in METHODS}
    for method in METHODS:
        for ndim_val in ndims:
            ess_val = ess_ndims[ndim_val][method][0].values  # select_dim_idx=0
            ess_single_dim_ndim[method].append(float(ess_val))

    for i, method in enumerate(METHODS):
        label = ALGOS[method]
        ax2.plot(
            ndims,
            ess_single_dim_ndim[method],
            marker=markers[i],
            markersize=6,
            label=label,
            color=color_palette[i],
        )

    ax2.set_yscale("log")
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, ymax * 13)  # y_lim_factor for ndim plot
    ax2.set_xlabel("dimension $d$")
    ax2.set_ylabel("relative ESS (log)")
    ax2.set_xticks(ndims)
    ax2.set_xticklabels(ndims)

    for i, ax in enumerate((ax1, ax2), 65):
        ax.annotate(chr(i), xy=(0.01, 0.92), xycoords="axes fraction", fontsize=16)

    # Adjust layout and save
    fig.tight_layout()

    # Save the combined figure
    fig.savefig(
        "results/ess_curve.pdf",
        transparent=True,
        dpi=150,
    )
    fig.tight_layout()
    plt.show()
