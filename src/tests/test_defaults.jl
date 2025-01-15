using Test
include("../config/defaults.jl")
using .Defaults

@testset "Defaults Test" begin
    println(Defaults.DEFAULT_INPUT_SIZE)
end 

@test Defaults.DEFAULT_N_BEES == UInt16(3)