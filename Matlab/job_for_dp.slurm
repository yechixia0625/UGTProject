#!/bin/bash
#SBATCH --ntasks-per-node=6
#SBATCH --mem=16G
#SBATCH --job-name=nonIID_DP
#SBATCH --mail-user=cye9@sheffield.ac.uk

module load MATLAB/2023b

delta=1e-2
noise_scale=10
matlab -nodisplay -nosplash -r "delta=$delta; noise_scale=$noise_scale; nonIID_DP; exit"
