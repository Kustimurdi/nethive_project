#!/bin/bash
#SBATCH --job-name=sim_array
#SBATCH --output=/scratch/n/N.Pfaffenzeller/nikolas_nethive/hive_logs/output_%A_%a.log
#SBATCH --error=/scratch/n/N.Pfaffenzeller/nikolas_nethive/hive_logs/error_%A_%a.log
#SBATCH --array=1-REPLACE_ME
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ge69mer@mytum.de
#SBATCH --partition=th-ws
#SBATCH --export=all


# Load Julia (adjust as needed)
#module load julia/1.9.4

# Define path
PARAM_PATH=REPLACE_PARAM_PATH
PARAM_FILE=$(printf "args_%03d.txt" "$SLURM_ARRAY_TASK_ID")

project_path=/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive
main_path=/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/src/main.jl

# Run Julia simulation
julia --project=$project_path $main_path --args_file="${PARAM_PATH}/${PARAM_FILE}" \
    2>&1 | tee /scratch/n/N.Pfaffenzeller/nikolas_nethive/hive_logs/full_log_${SLURM_JOB_ID}.log
