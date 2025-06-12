## Script Usage Guide

This guide explains how to reproduce the numerical illustrations from our [paper](https://doi.org/10.48550/arXiv.2301.08056). We recommend downloading the precomputed results from [Zenodo](https://doi.org/10.5281/zenodo.15610462) and storing them in a *results* directory rather than running all computations from scratch, as many scripts require significant computation time.

### Script Descriptions

#### 1. Bingham Distribution
- **Sampler**: `bingham.py`
- **Analysis**: `Bingham.ipynb` (or exported `Bingham.pdf`)

The notebook loads precomputed results and provides plotting tools and discussion of the Bingham distribution experiments.

#### 2. Rigid Registration Example
- **Script**: `protein_reg3d3d.py`
- **Usage**:
   ```sh
   python scripts/protein_reg3d3d.py \
    --n_samples $N_SAMPLES \ # no. of samples (default 2000)
    --burnin $BURNIN \       # fraction of burnin, (default 0.2)
    --n_chains $N_CHAINS \   # no. of chains (default 200)
    --out_dir $OUT_DIR \     # directory to save results
    --n_jobs $N_JOBS         # parallelization (default -1, so all available CPUs)
   ```

#### 3. Mixture of von Mises-Fisher (vMF) Distribution
- **Visualization**: `visualize_mixture_vMF.ipynb` (loads precomputed results)
- **Sampling**: `mixture_vMF.py`
- **Usage**:
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
- **ESS plotting**: `ess_mixture_vMF.py`

#### 4. Curved vMF Density
- **Visualization**: `visualize_curve_vMF.ipynb` (loads precomputed results)
- **Sampling**: `curve_vMF.py`
- **Usage**:
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

### Notes
- Replace `$VARIABLE_NAME` placeholders with appropriate values for your experiments
- Consider starting with smaller parameter values to test your setup before running full-scale experiments
