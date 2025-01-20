using Test
using Flux
include("../definitions.jl")
include("../config/defaults.jl")

using .Definitions
using .Defaults

@testset "Test Bee struct" begin
    bee = Definitions.Bee(1) 

    @test !isnothing(bee) 
    @test bee.id == 1 
    @test bee.params_history[0] == Flux.params(bee.brain)
end


@testset "Test Hive struct" begin
    hive = Definitions.Hive()
    @test !isnothing(hive)
    @test hive.n_bees == Defaults.DEFAULTS[:N_BEES]
end
