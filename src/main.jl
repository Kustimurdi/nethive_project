#!/bin/bash
#=
exec julia --optimize=3 --threads=4 "${BASH_SOURCE[0]}" "$@"
=#

import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")
Pkg.instantiate()

print(string("Current working dir: ", pwd(), "\n"))

"""
-------------------------------------------------------------------
1. part - setup of arguments and constants
-------------------------------------------------------------------
"""

include("helper.jl")
include("config/defaults.jl")
include("config/arg_table.jl")
include("config/constants.jl")
@show ARGS
save_params(parsed_args, RAW_PATH)


"""
-------------------------------------------------------------------
2. part - setup of the simulation and run 
-------------------------------------------------------------------
"""

include("definitions.jl")
include("prepare_data.jl")
include("methods.jl")


#include("tests/methods_for_testing.jl")
run_regression_sbatch(10000, 1000)
#run_straightuptrain(10000, 1000)
#run_testing_sbatch(10000, 1000)


@info string("DONE!")



"""
TODO:
- test the neural networks
- test the gillespie only with the neural networks learning DONE
- implement the rest of gillespie


- implement R code to load data and create plots for the individual bees and the average over the bees
- implement second training loop and adjust accurcies


- look at the ratio between the two propensities over time and adjust lambda_train accordingly
- adjust the @info messages

- propensity ratio has to be saved!!
- check learning process, somehow in the gillespie they learn faster than outside although atol is 
  even lower in gillespie
- using defaults results in problems sometimes when defining a value manually at points
"""