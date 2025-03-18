#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=16G
#SBATCH --job-name=nonIID_Simplex
#SBATCH --mail-user=cye9@sheffield.ac.uk
#SBATCH --array=1-6

module load MATLAB/2023b

echo "Running on node: $(hostname)"
matlab -nodisplay -nosplash -r "drop_client_ids=[$SLURM_ARRAY_TASK_ID]; nonIID_Simplex_Learning; exit"