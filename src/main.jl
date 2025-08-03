#!/bin/bash
#=
exec julia --optimize=3 --threads=4 "${BASH_SOURCE[0]}" "$@"
=#

# -- Environment setup --
import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")
#Pkg.instantiate()

print(string("Current working dir: ", pwd(), "\n"))


# -- Load packages in dependency-safe order--

# 1. Dependencies and general helpers
include("../src/dependencies.jl")
include("../src/helper.jl")  # Uses the packages defined above

# 2. Config and argument parsing
include("../src/config/defaults.jl")
include("../src/config/arg_table.jl")
include("../src/config/parse_args.jl")

# 3. Task system
include("../src/tasks/task_types.jl")     # AbstractTaskConfig, Task, etc.
include("../src/tasks/task_utils.jl")     # TaskConfig, accuracy, loss, etc.

# 4. Models
include("../src/models/models.jl")        # build_model, etc.

# 5. Core simulation logic
include("../src/core/definitions.jl")     # Bee, Hive, HiveConfig, etc.
include("../src/core/queen_gene.jl")      # QueenGeneMethod types and compute logic
include("../src/core/methods.jl")         # punish_model, gillespie_simulation, etc.
include("../src/tasks/task_training.jl")  # Training-related functions

# 6. Simulation entry point
include("../src/simulation.jl")


# -- Parse Args and Configure Simulation --
s = create_arg_parse_settings(DEFAULTS)
parsed_args = parse_args_with_args_file(s)
#parsed_args_tmp = parse_args(s)
#if parsed_args_tmp["args_file"] != ""
#  arg_lines = readlines(parsed_args_tmp["args_file"])
#  push!(Base.ARGS, arg_lines...)
#end
#
#parsed_args = parse_args(s)
@show ARGS


# -- Run Simulation --
hive = run_simulation(parsed_args; save_data=true, verbose=true, seed=parsed_args["random_seed"])
@info string("DONE!")


"""
TODO:
- test the neural networks


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