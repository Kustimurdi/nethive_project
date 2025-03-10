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


"""
-------------------------------------------------------------------
2. part - setup of the simulation and run 
-------------------------------------------------------------------
"""

include("definitions.jl")
include("methods.jl")

run_gillespie(n_epochs=N_EPOCHS, n_steps_per_epoch=N_STEPS_PER_EPOCH)
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
- implement another form of punishment:
    save old version of NN for all bees
    when in subdominant interaction -> change weights back to old set of weights
    problem: what if it gets dominated multiple times after another? if we only save the weights for one previous step, then it cannot be punished anymore after one subdominant interaction
"""