using Test
using Flux
include("../definitions.jl")
include("../config/defaults.jl")

@testset "Test Bee struct" begin
    bee = Bee(1) 

    @test !isnothing(bee) 
    @test bee.id == 1 
    @test bee.params_history[0] == Flux.params(bee.brain)
end


@testset "Test Hive struct" begin
    hive = Hive()
    @test !isnothing(hive)
    @test hive.n_bees == DEFAULTS[:N_BEES]
end

hive2 = Hive(brain_constructor=build_cifar10_model_small)