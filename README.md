<h1 align="center">
<img src="https://github.com/ShantanuKodgirwar/geosss/blob/main/assets/logo.svg" width="300">
</h1><br>

[![arXiv](https://img.shields.io/badge/DOI-10.1038%2Fs41586--020--2649--2-blue)](
https://doi.org/10.48550/arXiv.2301.08056)

# GeoSSS: Geodesic Slice Sampling on the Sphere

This Python package implements two new tuning-free MCMC algorithms **Rejection Spherical Slice Sampler** and **Shrinkage Spherical Slice Sampler** to sample from spherical distributions on arbitrary dimensions.

As shown in our [paper](https://doi.org/10.48550/arXiv.2301.08056), our algorithms have outperformed standard MCMC algorithms such as random-walk Metropolis-Hastings (RWMH) and Hamiltonian Monte Carlo (HMC) for spherical distributions. The package also includes the implementation of RWMH and a fine-tuned HMC.  

Example: Comparing the histogram of these samplers for a 10-dimensional mixture of von Mises-Fisher distribution. For more details on this example, please check the Numerical Illustrations section of the [paper](https://doi.org/10.48550/arXiv.2301.08056).


## Installation

GeoSSS is available for installation from PyPI (TODO: Add a link here later). Therefore, simply type:
```
pip install geosss
```

## Usage

To start using the four samplers in the package, simply do the following:
```python

from geosss import (RejectionSphericalSliceSampler, ShrinkageSphericalSliceSampler, 
                    MetropolisHastings, SphericalHMC)
# TODO: Need to finish example here

```

## Development

The package is maintained by [Poetry](https://python-poetry.org/). To install this package and its dependencies in a dedicated virtual environment using poetry, please do the following

```
git clone https://github.com/ShantanuKodgirwar/geosss
cd geosss
poetry install
```

To install the package in a development mode, please follow the below steps:

```
git clone https://github.com/ShantanuKodgirwar/geosss
cd geosss
pip install -e .
```

## Citation

If you use this package, or ideas from the paper, please consider citing us.
```
@misc{habeck2023,
      title={Geodesic slice sampling on the sphere}, 
      author={Michael Habeck and Mareike Hasenpflug and Shantanu Kodgirwar and Daniel Rudolf},
      year={2023},
      eprint={2301.08056},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```

