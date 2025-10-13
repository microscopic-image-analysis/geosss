#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name="ess_mixture_vMF"
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=job_logs/%j_%x_%N.log
#SBATCH --export=ALL

# This script runs the ESS analysis and generates plots for mixture vMF samplers
# The parameters (d=10, K=5, kappas=50-500) are hardcoded in the Python script

echo "Starting ESS analysis for mixture vMF samplers"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"

# Run the ESS analysis script
poetry run python scripts/ess_mixture_vMF.py

echo "ESS analysis completed"
