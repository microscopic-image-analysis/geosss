import os

import arviz as az
import numpy as np
from csb.io import dump, load

METHODS = ("sss-reject", "sss-shrink", "rwmh", "hmc")
ALGOS = {
    "sss-reject": "geoSSS (reject)",
    "sss-shrink": "geoSSS (shrink)",
    "rwmh": "RWMH",
    "hmc": "HMC",
}


def load_samples_varying_kappa(path_varying_kappa, kappas, n_dim=10, n_runs=10):
    """Load chains of samples for varying kappas. The shape is (n_runs, n_samples, n_dim).

    Parameters
    ----------
    path_varying_kappa : str
        subdirectory for each kappa consisting of samples per run
    kappas : list[float]
        concentration parameters
    n_dim : int
        dimensions of the samples, by default 10
    n_runs : int, optional
        no. of chains, by default 10

    Returns
    -------
    dict
        loads a dictionary that can be read as dict[kappa][method]
    """
    # Initialize a dictionary to hold data for each kappa and method
    data_dict = {kappa: {method: [] for method in METHODS} for kappa in kappas}

    for kappa in kappas:
        subdir = f"curve_{n_dim}d_kappa_{int(kappa)}"
        for chain in range(n_runs):
            samples_file = f"{path_varying_kappa}/{subdir}/curve_samples_{n_dim}d_kappa_{float(kappa)}_run{chain}.pkl"
            if os.path.exists(samples_file):
                samples_all = load(samples_file)
                # Add to the main dictionary
                for method in METHODS:
                    if method in samples_all:
                        data_dict[kappa][method].append(samples_all[method])
            else:
                error_msg = (
                    f"Make sure samples are precomputed and stored for {samples_file}"
                )
                raise FileNotFoundError(error_msg)

    # Optionally convert lists to arrays if needed
    for kappa in kappas:
        for method in data_dict[kappa].keys():
            data_dict[kappa][method] = np.stack(data_dict[kappa][method], axis=0)

    # samples stored as (chains, draws, dimensions) and accessed with data_dict[kappa][method]
    return data_dict


def load_samples_varying_ndim(path_varying_ndim, n_dims, kappa=500, n_runs=10):
    """Load chains of samples for varying kappas. The shape is (n_runs, n_samples, n_dim).

    Parameters
    ----------
    path_varying_ndim : str
        subdirectory for each kappa consisting of samples per run
    n_dims : list[int]
        list consisting of dimensions
    kappa : float, optional
        fixed concentration parameter, by default 500
    n_runs : int, optional
        no. of chains, by default 10

    Returns
    -------
    dict
        loads a dictionary that can be read as dict[n_dim][method]
    """
    # Initialize a dictionary to hold data for each kappa and method
    data_dict = {n_dim: {method: [] for method in METHODS} for n_dim in n_dims}

    for n_dim in n_dims:
        subdir = f"curve_{n_dim}d_kappa_{float(kappa)}"
        for chain in range(n_runs):
            samples_file = f"{path_varying_ndim}/{subdir}/curve_samples_{n_dim}d_kappa_{float(kappa)}_run{chain}.pkl"
            if os.path.exists(samples_file):
                samples_all = load(samples_file)
                # Add to the main dictionary
                for method in METHODS:
                    if method in samples_all:
                        data_dict[n_dim][method].append(samples_all[method])
            else:
                error_msg = (
                    f"Make sure samples are precomputed and stored for {samples_file}"
                )
                raise FileNotFoundError(error_msg)

    # Optionally convert lists to arrays if needed
    for n_dim in n_dims:
        for method in data_dict[n_dim].keys():
            data_dict[n_dim][method] = np.stack(data_dict[n_dim][method], axis=0)

    # samples stored as (chains, draws, dimensions) and accessed with data_dict[n_dim][method]
    return data_dict


def calc_ess(samples_dict, verbose=False):
    """
    ESS is calculated for every method with the default 'bulk' method from arviz.
    This implementation estimates ess values per dimension using N chains.

    Parameters
    ----------
    samples_dict : dict
        samples_dict[method] corresponds to samples for every method with shape (chains, draws, dimensions)
    verbose : bool

    Returns
    -------
    Xarray.Dataset
        Returns values as an Xarray datatset with ess values per dimension per method and can be
        access with `ess_vals[method].values`
    """

    # Converts the dict for a specific config (fixed kappa) to an arviz dataset
    samples_az = az.dict_to_dataset(samples_dict)

    # ensure chains are greater than or equal to 10
    for method in METHODS:
        chains = samples_az[method].values.shape[0]

        error_msg = f"ESS values not computed, either runs_samples is not an array or requires chains >= 10"
        assert chains >= 10

    # calculate ess when there are greater than or equal to 10 chains
    if isinstance(samples_dict, dict):
        # print(f"Calculating ESS from samples for kappa={kappa} and ndim={ndim}..")
        ess_vals = {method: None for method in METHODS}

        # estimates ESS per dimension for every method
        for method in METHODS:
            # samples from all runs with shape (chains, draws, dimensions)
            samples_per_method = samples_az[method]
            ess_vals[method] = az.ess(samples_per_method, relative=True)[method]

            # print ESS value per dimension
            if verbose:
                for i, vals in enumerate(ess_vals[method].values):
                    print(f"{method} ESS dim {i+1}: {vals:.4%}")

    else:
        error_msg = f"ESS values not computed, either runs_samples is not an array or requires chains >= 10"
        raise ValueError(error_msg)

    return ess_vals


def calc_ess_varying_kappa(
    kappas,
    subdir=None,
    ess_filename="ess_curve_varying_kappa.pkl",
    n_dim=10,
    n_runs=10,
    recompute_ess=False,
    verbose=False,
    return_ess=False,
):
    """Either loads from memory or calculates ESS for varying kappas (assumes 10 chains per kappa are precomputed).

    Parameters
    ----------
    kappas : list[float]
        varying concentration parameters
    subdir : str, optional
        Subdirectory consisting of all the files corresponding to varying kappas, by default None
    ess_filename : str, optional
        Filename for storing ESS values, by default "ess_curve_varying_kappa.pkl"
    n_dim : int, optional
        number of dimensions, by default 10
    n_runs : int, optional
        number of chains, by default 10
    recompute_ess : bool, optional
        whether or not to recompute ESS calculations, by default False
    return_ess : bool, optional
        whether or not to return ESS values, by default False
    """

    # loads samples for varying kappa as a dictionary with shape (chains, draws, dimensions) and
    # accessed with data_dict[kappa][method]
    subdir = (
        f"results/curve_{n_dim}d_vary_kappa_nruns_{n_runs}"
        if subdir is None
        else subdir
    )
    datasets_varying_kappa = load_samples_varying_kappa(
        subdir,
        kappas,
        n_dim=n_dim,
        n_runs=n_runs,
    )

    ess_filepath = os.path.join(subdir, ess_filename)
    if recompute_ess or not os.path.exists(ess_filepath):
        ess_kappas = {kappa: {} for kappa in kappas}
        for kappa in kappas:
            ess_kappas[kappa] = calc_ess(
                datasets_varying_kappa[kappa],
                verbose=verbose,
            )
        dump(ess_kappas, ess_filepath)
    else:
        ess_kappas = load(ess_filepath)

    return ess_kappas if return_ess else None


if __name__ == "__main__":

    kappas = np.arange(100, 900, 100)
    subdir = f"results/curve_10d_vary_kappa_nruns_10"
    ess_filename = "ess_curve_10d_varying_kappa.pkl"
    ess_kappas = calc_ess_varying_kappa(
        kappas,
        subdir,
        ess_filename,
        n_dim=10,
        n_runs=10,
        return_ess=True,
        recompute_ess=False,
    )
