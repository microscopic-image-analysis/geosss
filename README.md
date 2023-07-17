<h1 align="center">
<img src="https://github.com/ShantanuKodgirwar/geosss/blob/main/assets/logo.svg" width="300">
</h1><br>

[![arXiv](https://img.shields.io/badge/DOI-10.1038%2Fs41586--020--2649--2-blue)](
https://doi.org/10.48550/arXiv.2301.08056)

# GeoSSS: Geodesic Slice Sampling on the Sphere

This Python package implements two new tuning-free MCMC algorithms **Rejection Spherical Slice Sampler** and **Shrinkage Spherical Slice Sampler** to sample from spherical distributions on arbitrary dimensions.

As shown in our [paper](https://doi.org/10.48550/arXiv.2301.08056), our algorithms have outperformed standard MCMC algorithms such as random-walk Metropolis-Hastings (RWMH) and Hamiltonian Monte Carlo (HMC) for spherical distributions. The package also includes the implementation of RWMH and a fine-tuned HMC.  

Example: Comparing these samplers for a 10-dimensional mixture of von Mises-Fisher distribution. For more details on this example, please check the [paper](https://doi.org/10.48550/arXiv.2301.08056)

![mixture_vMF_d10_K5_kappa100_hist](https://github.com/ShantanuKodgirwar/geosss/assets/64919085/3d4fdd20-6157-4f81-9979-11d851c63daa)

