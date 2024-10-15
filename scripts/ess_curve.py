import os

import numpy as np
from csb.io import load


def load_samples_varying_kappa(methods, path, kappas, n_dim=10, n_runs=10):
    """Load chains of samples for varying kappas. The shape is (n_runs, n_samples, n_dim).

    Parameters
    ----------
    methods : str
        methods for which to load samples
    n_dim : int
        dimensions of the samples
    path : str
        main path
    kappas : list[float]
        concentration parameters
    n_runs : int, optional
        no. of chains, by default 10

    Returns
    -------
    dict
        loads a dictionary that can be read as dict[kappa][method]
    """
    # Initialize a dictionary to hold data for each kappa and method
    data_dict = {kappa: {method: [] for method in methods} for kappa in kappas}

    for kappa in kappas:
        subdir = f"curve_{n_dim}d_kappa_{int(kappa)}"
        for chain in range(n_runs):
            samples_file = f"{path}/{subdir}/curve_samples_{n_dim}d_kappa_{float(kappa)}_run{chain}.pkl"
            if os.path.exists(samples_file):
                samples_all = load(samples_file)
                # Add to the main dictionary
                for method in methods:
                    if method in samples_all:
                        data_dict[kappa][method].append(samples_all[method])

    # Optionally convert lists to arrays if needed
    for kappa in kappas:
        for method in data_dict[kappa].keys():
            data_dict[kappa][method] = np.stack(data_dict[kappa][method], axis=0)

    # samples stored as (chains, draws, dimensions) and accessed with data_dict[kappa][method]
    return data_dict


def load_samples_varying_ndim(methods, n_dims, path, kappa=500, n_runs=10):
    """Load chains of samples for varying kappas. The shape is (n_runs, n_samples, n_dim).

    Parameters
    ----------
    methods : str
        methods for which to load samples
    n_dim : int
        dimensions of the samples
    path : str
        main path
    kappas : list[float]
        concentration parameters
    n_runs : int, optional
        no. of chains, by default 10

    Returns
    -------
    dict
        loads a dictionary that can be read as dict[n_dim][method]
    """
    # Initialize a dictionary to hold data for each kappa and method
    data_dict = {n_dim: {method: [] for method in methods} for n_dim in n_dims}

    for n_dim in n_dims:
        subdir = f"curve_{n_dim}d_kappa_{float(kappa)}"
        for chain in range(n_runs):
            samples_file = f"{path}/{subdir}/curve_samples_{n_dim}d_kappa_{float(kappa)}_run{chain}.pkl"
            if os.path.exists(samples_file):
                samples_all = load(samples_file)
                # Add to the main dictionary
                for method in methods:
                    if method in samples_all:
                        data_dict[n_dim][method].append(samples_all[method])

    # Optionally convert lists to arrays if needed
    for n_dim in n_dims:
        for method in data_dict[n_dim].keys():
            data_dict[n_dim][method] = np.stack(data_dict[n_dim][method], axis=0)

    # samples stored as (chains, draws, dimensions) and accessed with data_dict[n_dim][method]
    return data_dict


if __name__ == "__main__":

    methods = ("sss-reject", "sss-shrink", "rwmh", "hmc")
    algos = {
        "sss-reject": "geoSSS (reject)",
        "sss-shrink": "geoSSS (shrink)",
        "rwmh": "RWMH",
        "hmc": "HMC",
    }

    kappas = np.arange(100, 900, 100)
    path = f"results/curve_10d_vary_kappa_nruns_10"
    datasets_varying_kappa = load_samples_varying_kappa(
        methods,
        path,
        kappas,
        n_dim=10,
        n_runs=10,
    )

    kappa = 500.0
    ndims = np.arange(3, 27, 3)
    path = f"results/curve_kappa_{float(kappa)}_vary_ndim_nruns_10/"
    datasets_varying_ndim = load_samples_varying_ndim(
        methods,
        ndims,
        path,
        kappa=500.0,
        n_runs=10,
    )
