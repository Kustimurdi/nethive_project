using Test
include("../definitions.jl")
include("../config/defaults.jl")

using .Definitions
using .Defaults

@testset "Test Bee struct" begin
    bee = Definitions.Bee(1) 

    # Test that bee is not nothing
    @test !isnothing(bee) 
    
    # Test that the id of the bee is 1
    @test bee.id == 1 
end

@testset "Test Hive struct" begin
    hive = Definitions.Hive()
    @test !isnothing(hive)
    @test hive.n_bees == Defaults.DEFAULT_N_BEES
end


