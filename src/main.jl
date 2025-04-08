#!/bin/bash
#=
exec julia --optimize=3 --threads=4 "${BASH_SOURCE[0]}" "$@"
=#

# -- Environment setup --
import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")
Pkg.instantiate()

print(string("Current working dir: ", pwd(), "\n"))

# -- Load packages --
include("dependencies.jl")
include("config/defaults.jl")
include("config/arg_table.jl")
include("config/parse_args.jl")
include("tasks/task_types.jl")
include("tasks/task_utils.jl")
include("tasks/task_training.jl")
include("core/definitions.jl")
include("core/methods.jl")
include("core/queen_gene.jl")
include("models/models.jl")
include("helper.jl")
include("simulation.jl")

# -- Parse Args and Configure Simulation --
s = create_arg_parse_settings(DEFAULTS)
parsed_args = parse_args(s)
@show ARGS


# -- Run Simulation --
run_simulation(parsed_args; save_results=true, verbose=true, seed=parsed_args["random_seed"])
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




"""
TODO2
- break circular dependency of queen gene und definitions
-  uberarbeite gillespie simulation sodass task type und queen gene am anfng in ein struct uebersetzt wird
"""