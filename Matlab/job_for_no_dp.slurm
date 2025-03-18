#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=16G
#SBATCH --job-name=nonIID_MOON_Main
#SBATCH --mail-user=cye9@sheffield.ac.uk

module load MATLAB/2023b

matlab -nodisplay -nosplash -r "nonIID_MOON_Main"
