#!/bin/bash
#=
exec julia --optimize=3 --threads=4 "${BASH_SOURCE[0]}" "$@"
=#

import Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")

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

println("Setup complete")

"""
-------------------------------------------------------------------
2. part - setup of the simulation and run 
-------------------------------------------------------------------
"""

include("definitions.jl")
include("methods.jl")
h = Hive(N_BEES, N_EPOCHS)
data_mnist = prepare_MNIST(false, false)
train_task!(h, data_mnist, N_EPOCHS)

@info string("DONE!")



"""
TODO:
- implement R code to load data and create plots for the individual bees and the average over the bees
- implement second training loop and adjust accurcies
"""