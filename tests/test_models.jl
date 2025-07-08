# src/tests/test_models.jl
using Test
using Flux
include("../src/models/models.jl")
include("../src/tasks/task_types.jl")  # Assuming RegressionTask etc. are here
include("../src/config/defaults.jl")  # For DEFAULTS

@testset "Model builder functions" begin
    # Sin model
    model = build_model_sin()
    @test model isa Chain
    @test size(model(rand(Float32, 1, 10))) == (1, 10)

    # Linear model
    model = build_model_linear()
    @test model isa Chain
    @test size(model(rand(Float32, 1, 10))) == (1, 10)

    # Sin model with leaky relu
    model = build_model_sin_leaky()
    @test model isa Chain
    @test size(model(rand(Float32, 1, 10))) == (1, 10)

    model = build_custom_classification_model(10, 5)
    @test model isa Chain

    # Dispatch tests
    @test build_model(RegressionTask()) isa Chain
    @test build_model(LinearRegressionTask()) isa Chain
    @test build_model(CustomClassificationTask()) isa Chain
    @test build_model(NoTask()) isa Chain
end
