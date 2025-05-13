#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --job-name=nonIID_MOON_Main
#SBATCH --mail-user=cye9@sheffield.ac.uk

module load MATLAB/2023b

matlab -nodisplay -nosplash -r "IID_FedAvg_Main"
