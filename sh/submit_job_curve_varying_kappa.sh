#!/bin/bash

# configs
export N_SAMPLES=1000000
export BURNIN=0.2
export N_CHAINS=10
export DIMENSION=5
export OUT_DIR_BASE="results"

# loop over kappas from 100 to 800 in steps of 100
for CONCENTRATION in $(seq 100.0 100.0 800.0)
do
    # export the current kappa to the environment
    export CONCENTRATION

    # export the output directory
    SUBDIR="curve_${DIMENSION}d_vary_kappa_nruns_${N_CHAINS}"
    OUT_DIR="${OUT_DIR_BASE}/${SUBDIR}/curve_${DIMENSION}d_kappa_${CONCENTRATION}"
    export OUT_DIR

    # Submit the SLURM job, passing all environment variables
    sbatch --export=ALL --job-name="curve_${DIMENSION}d_kappa${CONCENTRATION}_nruns${N_CHAINS}" sh/parallelized_job_curve.sh
done
