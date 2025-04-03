
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

include("../src/helper.jl")
include("../src/config/defaults.jl")
include("../src/config/arg_table.jl")
include("../src/config/constants.jl")
@show ARGS
save_params(parsed_args, RAW_PATH)


"""
-------------------------------------------------------------------
2. part - setup of the simulation and run 
-------------------------------------------------------------------
"""

include("../src/definitions.jl")
include("../src/methods.jl")

model = build_model_5()
run_training(model, dataset_function=prepare_cifar10_dataset_greyscale)

@info string("DONE!")


