using Test
include("../src/core/definitions.jl")
include("../src/tasks/task_types.jl")
include("../src/tasks/task_utils.jl")

@testset "HiveConfig Creation" begin
    args = Dict(
        "parent_dataset_name" => "dummy_parent",
        "n_bees" => 5,
        "n_epochs" => 3,
        "n_steps_per_epoch" => 2,
        "learning_rate" => 0.01,
        "punish_rate" => 0.1f0,
        "lambda_train" => 0.05,
        "lambda_interact" => 0.03,
        "accuracy_sigma" => 0.01,
        "random_seed" => 42,
        "task_type" => :regression,
        "queen_gene_method" => :accuracy,
        "regression_n_peaks" => 3,
        "regression_which_peak" => 1,
        "trainset_size" => 100,
        "testset_size" => 50,
        "initial_queen_gene" => Float16(0.5)
    )

    config = create_hive_config(args)
    
    @test config isa HiveConfig
    @test config.n_bees == 5
    @test config.learning_rate > 0
    @test config.task_config isa RegressionTaskConfig
end

@testset "Bee creation and brain" begin
    task = RegressionTask()
    task2 = NoTask()
    bee = Bee(1, task, 0.3)
    bee2 = Bee(2, task2, 0.5)
    @test bee isa Bee
    @test bee2 isa Bee 

    @test bee.id == 1
    @test bee.queen_gene ≈ 0.3
    @test bee2.queen_gene ≈ 0.5
    @test bee.brain isa Flux.Chain
    @test bee2.brain isa Flux.Chain
end

@testset "Hive creation from config" begin
    args = Dict(
        "parent_dataset_name" => "dummy_parent",
        "n_bees" => 3,
        "n_epochs" => 2,
        "n_steps_per_epoch" => 1,
        "learning_rate" => 0.01,
        "punish_rate" => 0.1f0,
        "lambda_train" => 0.05,
        "lambda_interact" => 0.03,
        "accuracy_sigma" => 0.01,
        "random_seed" => 123,
        "task_type" => :regression,
        "queen_gene_method" => :accuracy,
        "regression_n_peaks" => 2,
        "regression_which_peak" => 1,
        "trainset_size" => 100,
        "testset_size" => 50
    )

    config = create_hive_config(args)
    hive = Hive(config)

    @test hive isa Hive
    @test length(hive.bee_list) == config.n_bees
    @test size(hive.loss_history) == (config.n_bees, config.n_epochs)
    @test all(hive.initial_accuracies_list .== 0.0)
    @test hive.propensity_ratio_history == fill(-1.0, config.n_epochs)
end

@testset "HivePaths creation" begin
    args = Dict(
        "parent_dataset_name" => "dummy_parent",
        "n_bees" => 2,
        "n_epochs" => 1,
        "n_steps_per_epoch" => 1,
        "learning_rate" => 0.01,
        "punish_rate" => 0.1f0,
        "lambda_train" => 0.05,
        "lambda_interact" => 0.03,
        "accuracy_sigma" => 0.01,
        "random_seed" => 1,
        "task_type" => :none,
        "queen_gene_method" => :accuracy
    )

    config = create_hive_config(args)
    paths = create_hive_paths(config)

    @test paths isa HivePaths
    @test occursin(config.dataset_name, paths.raw_path)
end
