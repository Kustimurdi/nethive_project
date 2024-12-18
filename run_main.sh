#!/usr/bin/env bash

module unload julia
module load julia/1.11.1

echo $PWD
echo "$@"
julia --project=./env_nethive ./src/main.jl "$@"

exit 0
