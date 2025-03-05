#!/usr/bin/env bash
#SBATCH --output=../logs/output_%j.log
#SBATCH --error=../logs/error_%j.log
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=6GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ge69mer@mytum.de
#SBATCH --export=all
#SBATCH --partition=th-ws

date
echo $PWD

echo $@
julia --project=/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/src/tests/testing.jl $@ 2>&1 | tee /scratch/n/N.Pfaffenzeller/nikolas_nethive/logs/full_log_${SLURM_JOB_ID}.log

date
exit 0
