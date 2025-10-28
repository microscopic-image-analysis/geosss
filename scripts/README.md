## Script Usage Guide

This guide explains how to reproduce the numerical illustrations from our [paper](https://doi.org/10.48550/arXiv.2301.08056). We recommend downloading the precomputed results from [ScienceDB](https://doi.org/10.57760/sciencedb.30181) and storing them in a *results* directory rather than running all computations from scratch, as many scripts require significant computation time.

### Script Descriptions

#### 1. Bingham Distribution
- **Sampling**: `bingham.py` (configurable via in-script parameters)
- **Analysis**: `Bingham.ipynb` (or exported `Bingham.pdf`)
- **Usage**:
   ```sh
   python scripts/bingham.py
   ```
   Note: Parameters (dimension, eigenvalues, number of samples, etc.) are configured within the script rather than via command-line arguments.

The notebook loads precomputed results and provides plotting tools and discussion of the Bingham distribution experiments.

#### 2. Rigid Registration of Biomolecular Structures
This demonstrates the implementation of section 4.1 in the paper.  
- **Sampling**: `protein_reg3d3d.py`
- **Usage**:
   ```sh
   python scripts/protein_reg3d3d.py \
    --n_samples $N_SAMPLES \  # no. of samples (default 2000)
    --burnin $BURNIN \        # fraction of burnin (default 0.2)
    --n_chains $N_CHAINS \    # no. of chains (default 200)
    --out_dir $OUT_DIR \      # directory to save results
    --n_jobs $N_JOBS          # parallelization (default -1, so all available CPUs)
   ```

#### 3. Mixture of von Mises-Fisher (vMF) Distribution
This demonstrates the section 4.2 in the paper where tests run our slice samplers, HMC and RWMH as well as a custom mixture proposal Metropolis algorithm.
- **Visualization**: `visualize_mixture_vMF.ipynb` (loads precomputed results)
- **Sampling**:
  - `mixture_vMF.py` - General mixture vMF sampler
  - `mixture_vMF_rwmh_indep.py` - Mixture proposal metropolis algorithm
- **Usage** (`mixture_vMF.py`):
   ```sh
   python scripts/mixture_vMF.py \
    --n_samples $N_SAMPLES \
    --burnin $BURNIN \
    --n_chains $N_CHAINS \
    --dimension $DIMENSION \          # dimensionality of the distribution
    --components $COMPONENTS \        # no. of vMF mixture components (default 5)
    --concentration $CONCENTRATION \  # concentration parameter of vMF
    --out_dir $OUT_DIR
   ```
- **Usage** (`mixture_vMF_rwmh_indep.py`):
   ```sh
   python scripts/mixture_vMF_rwmh_indep.py \
    --n_samples $N_SAMPLES \
    --burnin $BURNIN \
    --n_chains $N_CHAINS \
    --dimension $DIMENSION \
    --components $COMPONENTS \
    --concentration $CONCENTRATION \
    --mix_prob $MIX_PROB \            # mixing probability for independence sampler
    --out_dir $OUT_DIR
   ```
- **ESS plotting**: `ess_mixture_vMF.py`

#### 4. Curved Distribution on the Sphere
This implements the Curved vMF distribution on the sphere as discussed in section 4.3 of the paper. Additionally, a note on the gradient calculation for HMC is added under `gradient-curve.pdf`
- **Visualization**: `visualize_curve_vMF.ipynb` (loads precomputed results)
- **Sampling**:
  - `curve_vMF.py` - General curved vMF sampler
  - `curve_3d.py` - Specialized 3D case comparing samplers on curves on a 2-sphere
- **Usage** (`curve_vMF.py`):
   ```sh
   python scripts/curve_vMF.py \
    --n_samples $N_SAMPLES \
    --burnin $BURNIN \
    --n_chains $N_CHAINS \
    --dimension $DIMENSION \
    --concentration $CONCENTRATION \
    --out_dir $OUT_DIR
   ```
- **ESS plotting**: `ess_curve_vMF.py`

### Utility Scripts

Remaining scripts are mainly utility or plotting scripts

### Notes
- Replace `$VARIABLE_NAME` placeholders with appropriate values for your experiments
- Consider starting with smaller parameter values to test your setup before running full-scale experiments
