#!/usr/bin/env bash

echo $PWD
echo "$@"
julia --project=./env_nethive ./src/main.jl "$@"

exit 0
