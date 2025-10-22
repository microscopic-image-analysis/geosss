#!/bin/bash

# configs
export N_SAMPLES=1000000
export BURNIN=0.2
export N_CHAINS=10
export DIMENSION=10
export OUT_DIR_BASE="results"
export COMPONENTS=5

# Nested loop: outer loop over kappas, inner loop over mixing probabilities
for CONCENTRATION in $(seq 50.0 50.0 500.0)
do
    # Loop over mixing probabilities from 0.1 to 1.0 in steps of 0.1
    for MIX_PROB in $(seq 0.1 0.1 1.0)
    do
        # Export the current kappa and mixing probability to the environment
        export CONCENTRATION
        export MIX_PROB

        # Export the output directory (no need to include curve_ prefix)
        export OUT_DIR="${OUT_DIR_BASE}"

        # Submit the SLURM job, passing all environment variables
        # Job name includes both kappa and mixing probability
        sbatch --export=ALL \
            --job-name="mix_d${DIMENSION}_kappa${CONCENTRATION}_mixprob${MIX_PROB}" \
            sh/parallelized_job_mixture_sampler.sh
    done
done
