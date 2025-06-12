<p align="center">
<img src="https://raw.githubusercontent.com/microscopic-image-analysis/geosss/927ff8c8187b88a1a72725c4e450ae0f0523431b/assets/logo.svg" width="300">
</p>

<div align="center">

  [![PyPI](https://img.shields.io/pypi/v/geosss)](https://pypi.org/project/geosss/)
  ![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)
  [![arXiv](https://img.shields.io/badge/DOI-10.1038%2Fs41586--020--2649--2-blue)](
  https://doi.org/10.48550/arXiv.2301.08056)
  [![License](https://img.shields.io/badge/License-BSD_3--Clause-purple.svg)](https://opensource.org/licenses/BSD-3-Clause)

</div>

# GeoSSS: Geodesic Slice Sampling on the Sphere

This python package implements two novel tuning-free MCMC algorithms to sample distributions defined on the sphere embedded in &#8477;<sup>d</sup>, by leveraging its underlying manifold geometry. The two samplers namely **ideal geodesic slice sampler** based on accept/reject strategy and a **shrinkage-based geodesic slice sampler** are implemented. The latter is a much faster, but slightly less accurate variant and therefore recommended for practical use due to its efficiency.  

In addition, the package also provides the implementation of the spherical variants of random-walk Metropolis-Hastings (RWMH) and Hamiltonian Monte Carlo (HMC). As demonstrated in our [paper](https://doi.org/10.48550/arXiv.2301.08056), the proposed samplers outperform RWMH and HMC for challenging target distributions. 

To reproduce the results in the paper, see this [section](#development-and-reproducibility). However, to get started quickly, install the package and follow along with the demo provided below. 


## Installation

GeoSSS is available for installation from [PyPI](https://pypi.org/project/geosss/). Therefore, simply type:

```bash
pip install geosss
```

## Minimal Example

We consider a target that is a mixture of von Mises-Fisher distributions on $\mathbb{S}^2$ with concentration parameter $\kappa=$ 80. By using 1000 samples, our samplers geoSSS (reject) and geoSSS (shrink) explore all modes, whereas RWMH and HMC get stuck in a single mode. 

<p align="center">
<img src="https://github.com/microscopic-image-analysis/geosss/blob/1ed528f2b708cfc8b88bd78bd8f210e6a0d6372a/assets/animation_vMF.gif" width="1000">
</p>

This demo can be created with the below script.
```python
import geosss as gs
import numpy as np

# Create mixture of von Mises-Fisher distributions
mus = np.array([[0.87, -0.37, 0.33],
                [-0.20, -0.89, -0.40],
                [0.19, 0.22, -0.96]])
vmfs = [gs.VonMisesFisher(80.0 * mu) for mu in mus]
pdf = gs.MixtureModel(vmfs)

# Sampling parameters
n_samples, burnin = 1000, 100
init_state = np.array([-0.86, 0.19, -0.47])
seed = 3521

# Sample with different methods
samplers = {
    'sss-reject': gs.RejectionSphericalSliceSampler, # very accurate, but slow
    'sss-shrink': gs.ShrinkageSphericalSliceSampler, # reasonably accurate, but fast
    'rwmh': gs.MetropolisHastings,                   # automatically tuned during burnin          
    'hmc': gs.SphericalHMC,                          # automatically tuned during burnin
}

samples = {name: cls(pdf, init_state, seed).sample(n_samples, burnin) 
           for name, cls in samplers.items()}
```
See the notebook [`demo.ipynb`](demo.ipynb) for visualization of the samples.

## Development and Reproducibility

To reproduce results from the numerical illustrations section of the paper, check the [scripts](scripts/) directory. Precomputed results can also be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.15610462) and plotted with these scripts.

However, first installing the package and it's *locked* dependencies is necessary and can be done as follows:

1. Clone the repository and navigate to the root of the folder,

```bash
git clone https://github.com/microscopic-image-analysis/geosss.git
cd geosss
```

2. You can now create a virtual environment (with `conda` for example),

```bash
conda create --name geosss-venv python=3.12 # or python >= 3.10, < 3.13
conda activate geosss-venv
```

3. The dependencies can be installed in this environment as,
```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Optionally, this can also be done with the python package manager [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) as

```bash
poetry sync --all-extras
```

## Citation

If you use this package or ideas from the paper, please consider citing us.
```bash
@misc{habeck2023,
      title={Geodesic slice sampling on the sphere}, 
      author={Michael Habeck and Mareike Hasenpflug and Shantanu Kodgirwar and Daniel Rudolf},
      year={2023},
      eprint={2301.08056},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```

