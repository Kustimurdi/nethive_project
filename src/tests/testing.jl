using Pkg
Pkg.activate("./env_nethive/")
using Test

include("../helper.jl")
include("../config/defaults.jl")
include("../config/arg_table.jl")
include("../config/constants.jl")

include("../definitions.jl")
include("../methods.jl")

using Images

## MLDatasets testing
x_fashion, y_fashion = MLDatasets.FashionMNIST(:train)[:]

index = 3
img = x_fashion[:, :, index]

colorview(Gray, img')