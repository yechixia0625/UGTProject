#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --job-name=nonIID_MOON_Main
#SBATCH --mail-user=cye9@sheffield.ac.uk
#SBATCH --array=6

module load MATLAB/2023b

echo "Running on node: $(hostname)"
matlab -nodisplay -nosplash -r "drop_client_ids=[$SLURM_ARRAY_TASK_ID]; nonIID_MOON_Dropout; exit"