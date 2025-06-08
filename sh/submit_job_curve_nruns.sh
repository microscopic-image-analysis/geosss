#!/bin/bash

# configuration
export N_SAMPLES=1000000
export BURNIN=0.2
export N_CHAINS=10
export DIMENSION=3
export CONCENTRATION=800.0

# Submit the SLURM job and pass the parameters as arguments
sbatch --export=ALL --job-name="curve_${DIMENSION}d_kappa${CONCENTRATION}_nruns${N_CHAINS}" sh/parallelized_job_curve.sh
