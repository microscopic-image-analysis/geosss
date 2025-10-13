#!/bin/bash

#SBATCH --partition=fat
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --output=job_logs/%j_%x_%N.log
#SBATCH --export=ALL

# Execute the Python script with parameters
# NOTE: See files `submit_job_*` for submitting multiple jobs. This one is 
# specifically for `sh/submit_job_mixture_sampler_vMF_varying_kappa`
poetry run python scripts/mixture_vMF_mixture_sampler.py \
    --n_samples $N_SAMPLES \
    --burnin $BURNIN \
    --n_chains $N_CHAINS \
    --dimension $DIMENSION \
    --components $COMPONENTS \
    --concentration $CONCENTRATION \
    --mix_prob $MIX_PROB \
    --out_dir $OUT_DIR