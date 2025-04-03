empty!(DEPOT_PATH)  # Clear the current path
push!(DEPOT_PATH, "/scratch/n/N.Pfaffenzeller/.julia")

using Pkg
Pkg.activate("./env_nethive/")
Pkg.precompile()
using Test

include("../helper.jl")
include("../config/defaults.jl")
include("../config/arg_table.jl")
include("../config/constants.jl")

include("../definitions.jl")
include("../methods.jl")

using Images

