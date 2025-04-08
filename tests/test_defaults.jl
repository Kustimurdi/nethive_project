using Test
include("../config/defaults.jl")

@testset "Defaults Test" begin
    println(DEFAULTS[:INPUT_SIZE])
end 

@test DEFAULTS[:N_BEES] == UInt16(3)