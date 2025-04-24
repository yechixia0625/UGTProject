#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=32G
#SBATCH --job-name=nonIID_MOON_DP
#SBATCH --mail-user=cye9@sheffield.ac.uk
module load MATLAB/2023b

echo "Running on node: $(hostname)"
matlab -nodisplay -nosplash -r "DPNoiseMult=1; nonIID_MOON_DP; exit"
