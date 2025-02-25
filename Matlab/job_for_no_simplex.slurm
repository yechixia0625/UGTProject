#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=no_simplex_learning

module load MATLAB/2023b

matlab -nodisplay -nosplash -r "nonIID_MOON_Dropout"
