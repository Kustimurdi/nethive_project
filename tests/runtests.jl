# Activate project environment
import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")
#Pkg.instantiate()

# === Load all required source files in dependency-safe order ===

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
include("../src/tasks/task_training.jl")  # Training-related functions

# 4. Core simulation logic
include("../src/core/queen_gene.jl")      # QueenGeneMethod types and compute logic
include("../src/core/definitions.jl")     # Bee, Hive, HiveConfig, etc.
include("../src/core/methods.jl")         # punish_model, gillespie_simulation, etc.

# 5. Models
include("../src/models/models.jl")        # build_model, etc.

# 6. Simulation entry point
include("../src/simulation.jl")

# === Include test files ===

include("test_parse_args.jl")
include("test_helper.jl")
# include("test_definitions.jl")
# include("test_tasks.jl")
# include("test_methods.jl")
# include("test_simulation.jl")


"""
empty!(DEPOT_PATH)  # Clear the current path
push!(DEPOT_PATH, "/scratch/n/N.Pfaffenzeller/.julia")

using Pkg
Pkg.activate("./env_nethive/")
#Pkg.instantiate()
using Test

include("../src/dependencies.jl")
include("test_helper.jl")
#include("test_models.jl")
#include("test_methods.jl")
#include("test_tasks.jl")
#include("test_queen_gene.jl")
#include("test_definitions.jl")



include("../helper.jl")
include("../config/defaults.jl")
include("../config/arg_table.jl")
include("../config/constants.jl")

include("../definitions.jl")
include("../methods.jl")

using Images

"""
