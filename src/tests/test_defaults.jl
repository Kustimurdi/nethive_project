using Test
include("../config/defaults.jl")
using .Defaults

@testset "Defaults Test" begin
    println(Defaults.DEFAULTS[:INPUT_SIZE])
end 

@test Defaults.DEFAULTS[:N_BEES] == UInt16(3)