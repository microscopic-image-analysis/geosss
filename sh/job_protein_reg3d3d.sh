#!/bin/bash
#SBATCH --partition=fat
#SBATCH --job-name="protein_reg3d3d"
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=4G
#SBATCH --output=job_logs/%j_%x_%N.log
#SBATCH --export=ALL

# Set parameters
N_SAMPLES=2000
BURNIN=0.2 # returns all samples after burnin (adapts step-size during burnin)
N_CHAINS=200
OUT_DIR="results/protein_reg3d3d_CPD_chains_${N_CHAINS}_rerun" 
N_JOBS=${SLURM_CPUS_PER_TASK}

# Now run the Python script with the parallelized sampling
echo "Running with class_avg_idx=$CLASS_AVG_IDX"
echo "Using $N_JOBS parallel jobs for sampling"

# Execute the Python script with parameters
poetry run python scripts/protein_reg3d3d.py \
    --n_samples $N_SAMPLES \
    --burnin $BURNIN \
    --n_chains $N_CHAINS \
    --out_dir $OUT_DIR \
    --n_jobs $N_JOBS \