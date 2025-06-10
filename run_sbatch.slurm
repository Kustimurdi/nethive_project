#!/usr/bin/env bash
#SBATCH --output=../hive_logs/output_%j.log
#SBATCH --error=../hive_logs/error_%j.log
#SBATCH --open-mode=append
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=ge69mer@mytum.de
#SBATCH --export=all
#SBATCH --partition=th-ws
#SBATCH --export=ALL,DISPLAY=""

date
echo $PWD

echo $@
julia --project=/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/src/main.jl $@ 2>&1 | tee /scratch/n/N.Pfaffenzeller/nikolas_nethive/hive_logs/full_log_${SLURM_JOB_ID}.log

date
exit 0
