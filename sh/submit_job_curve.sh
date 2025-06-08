#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name="curve_vMF_3d"
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --output=job_logs/%j_%x_%N.log
#SBATCH --export=ALL

export N_SAMPLES=1000000
export BURNIN=0.2
export N_CHAINS=1
export DIMENSION=20
export CONCENTRATION=500.0

# Execute the Python script with parameters
# NOTE: This file will run independently, by submitting it directly to SLURM.
poetry run python scripts/curve_vMF.py \
    --n_samples $N_SAMPLES \
    --burnin $BURNIN \
    --n_chains $N_CHAINS \
    --dimension $DIMENSION \
    --concentration $CONCENTRATION \
    --generate_plots