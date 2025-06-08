#!/bin/bash

# configs
export N_SAMPLES=1000000
export CONCENTRATION=800.0
export N_RUNS=10
export OUT_DIR_BASE="results"
export BURNIN=0.2

# loop over dimensions from 3 to 24 in steps of 3
for DIMENSION in {3..24..3}
do
    # export the current dimension to the environment
    export DIMENSION

    # export the output directory
    SUBDIR="curve_kappa_${CONCENTRATION}_vary_ndim_nruns_${N_RUNS}"
    OUT_DIR="${OUT_DIR_BASE}/${SUBDIR}/curve_${DIMENSION}d_kappa_${CONCENTRATION}"
    export OUT_DIR
    
    # Submit the SLURM job, passing all environment variables
    sbatch --export=ALL --job-name="curve_${DIMENSION}d_kappa${CONCENTRATION}_nruns${N_RUNS}" sh/parallelized_job_curve.sh
done
