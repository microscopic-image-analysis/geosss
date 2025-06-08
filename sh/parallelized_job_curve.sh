#!/bin/bash

#SBATCH --partition=fat
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30G
#SBATCH --output=job_logs/%j_%x_%N.log
#SBATCH --export=ALL

# Execute the Python script with parameters
# NOTE: See files starting with submit_job_curve_ for submitting multiple jobs
poetry run python scripts/curve_vMF.py \
    --n_samples $N_SAMPLES \
    --burnin $BURNIN \
    --n_chains $N_CHAINS \
    --dimension $DIMENSION \
    --concentration $CONCENTRATION \
    --out_dir $OUT_DIR