#!/usr/bin/env bash
#SBATCH --output=../logs/output_%j.log
#SBATCH --error=../logs/error_%j.log
#SBATCH --open-mode=append
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END
#SBATCH --mail-user=ge69mer@mytum.de
#SBATCH --export=all

echo $PWD

echo $@
julia --project=/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/src/main.jl $@

exit 0
