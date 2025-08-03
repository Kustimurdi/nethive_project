#!/usr/bin/env julia

# Usage: julia submit_job_array.jl path/to/param_folder

import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")

using Printf

# === Get argument ===
if length(ARGS) != 1
    println("Usage: julia submit_job_array.jl path/to/param_folder")
    exit(1)
end

data_root = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params/"
param_folder = ARGS[1]
param_path = joinpath(data_root,param_folder)
param_files = filter(f -> endswith(f, ".txt"), readdir(param_path))
n_jobs = length(param_files)

if n_jobs == 0
    println("No .txt parameter files found in $param_folder")
    exit(1)
end

# === Read template ===
template_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/sbatch_array_template.sh"
template = read(template_path, String)

# === Replace placeholders ===
job_script = replace(template, 
    "REPLACE_ME" => string(n_jobs),
    "REPLACE_PARAM_PATH" => abspath(param_path)
)

# === Write temp script ===
script_path = joinpath(pwd(), "tmp_sbatch_array.sh")
open(script_path, "w") do io
    write(io, job_script)
end

# === Submit job ===
println("Submitting job array with $n_jobs tasks...")
run(`sbatch $script_path`)

rm(script_path, force=true)
println("Temporary job script deleted.")