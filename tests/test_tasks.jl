using Test
using Flux
using Random

include("../src/tasks/task_types.jl")
include("../src/tasks/task_utils.jl")
include("../src/tasks/task_training.jl")

@testset "Task Types and Configs" begin
    @test RegressionTask() isa Task
    @test LinearRegressionTask() isa Task
    @test ClassificationTask([28,28], 10) isa Task
    @test NoTask() isa Task

    config = RegressionTaskConfig(3, 1, 100, 50)
    @test config.n_peaks == 3
    @test config.which_peak == 1
    @test config.trainset_size == 100
    @test config.testset_size == 50
end

@testset "TaskConfig creation" begin
    @testset "TaskConfig creation - custom args" begin
        args = Dict(
            "task_type" => :regression,
            "regression_n_peaks" => 2,
            "regression_which_peak" => 1,
            "trainset_size" => 100,
            "testset_size" => 50
        )
        cfg = TaskConfig(args)
        @test cfg isa RegressionTaskConfig
    end
    @testset "TaskConfig creation - parsed args" begin
        settings = create_arg_parse_settings()
        args = cparse_args(settings)
        println(args)
        cfg = TaskConfig(args)
        println(cfg)
        @test cfg isa RegressionTaskConfig
        @test cfg.n_peaks == DEFAULTS[:REGRESSION_N_PEAKS]
        @test cfg.which_peak == DEFAULTS[:REGRESSION_WHICH_PEAK]
    end
    @testset "TaskConfig creation - no task type" begin
        args = Dict("task_type" => :none)
        cfg = TaskConfig(args)
        @test cfg isa NoTaskConfig
    end
    @testset "TaskConfig creation - invalid task type" begin
        args = Dict("task_type" => :invalid_task)
        @test_throws ArgumentError TaskConfig(args)
    end
end

@testset "Dataset creation" begin
    @testset "Regression dataset creation" begin
        cfg = RegressionTaskConfig(3, 2, 128, 64)

        # Test sin dataset
        train_loader, test_loader, train_data, test_data = create_dataset(RegressionTask(), cfg)
        @test size(train_data[1]) == (1, 128)
        @test size(train_data[2]) == (1, 128)
        @test size(test_data[1]) == (1, 64)
        @test size(test_data[2]) == (1, 64)
    end
    @testset "Linear dataset creation" begin
        cfg = RegressionTaskConfig(3, 1, 32, 16)
        train_loader, test_loader, train_data, test_data = create_dataset(LinearRegressionTask(), cfg)
        @test size(train_data[1]) == (1, 32)
        @test size(train_data[2]) == (1, 32)
    end
    @testset "No task dataset creation" begin
        cfg = NoTaskConfig()
        train_loader, test_loader, train_data, test_data = create_dataset(NoTask(), cfg)
        @test isnothing(train_loader)
        @test isnothing(test_loader)
        @test isnothing(train_data)
        @test isnothing(test_data)
    end
    @testset "Classification dataset creation" begin
        cfg = ClassificationTaskConfig(10, [0.1, 0.2, 0.3, 0.4])
        @test_throws ArgumentError create_dataset(ClassificationTask([28, 28], 10), cfg)
    end
end


@testset "Loss and accuracy functions" begin
    Random.seed!(42)  # for reproducibility

    model = Chain(Dense(1, 16, relu), Dense(16, 1))
    model2 = Chain(Dense(1, 16, relu), Dense(16, 1))
    cfg = RegressionTaskConfig(2, 1, 64, 32)
    train_loader, test_loader, train_data, test_data = create_dataset(RegressionTask(), cfg)

    loss = compute_task_loss(RegressionTask(), model, train_data[1], train_data[2])
    @test loss ≥ 0.0

    initial_loss = calc_loss(model, test_loader, RegressionTask())
    @test initial_loss ≥ 0.0

    acc = calc_accuracy(model, test_loader, RegressionTask())
    @test 0.0 ≤ acc ≤ 1.0

    train_model!(model, train_loader, RegressionTask())

    new_loss = calc_loss(model, test_loader, RegressionTask())
    @test new_loss < loss

    new_acc = calc_accuracy(model, test_loader, RegressionTask())
    @test new_acc > acc

    initial_loss2 = calc_loss(model2, test_loader, RegressionTask())
    @test initial_loss2 ≥ 0.0
    acc2 = calc_accuracy(model2, test_loader, RegressionTask())
    @test 0.0 ≤ acc2 ≤ 1.0
    punish_model!(model2, train_loader, RegressionTask())

    new_loss2 = calc_loss(model2, test_loader, RegressionTask())
    @test new_loss2 > initial_loss2
    new_acc2 = calc_accuracy(model2, test_loader, RegressionTask())
    @test new_acc2 < acc2
end

#sin_data = create_sin_dataset(3, 1, 1000)
#lin_data = create_linear_dataset(1000)
#plot_dataset(lin_data)
#model = build_model(RegressionTask())
include("../analysis/analysis_jl/extra/methods_repl.jl")
bee1 = Bee(1, RegressionTask())
model = bee1.brain
cfg = RegressionTaskConfig(6, 3, 1000, 500)
train_loader, test_loader, train_data, test_data = create_dataset(RegressionTask(), cfg)
acc = calc_accuracy(model, test_loader, RegressionTask())
loss = calc_loss(model, test_loader, RegressionTask())
println("Initial accuracy: ", acc)
println("Initial loss: ", loss)
plot_bee_prediction(model, train_data[1])
plot_dataset(train_data)
for i in 1:10000
    println("Training iteration: ", i)
    train_model!(model, train_loader, RegressionTask(), learning_rate=0.0001)
    acc = calc_accuracy(model, test_loader, RegressionTask(), acc_sigma=0.1)
    #loss = calc_loss(model, test_loader, RegressionTask())
    println("Accuracy: ", acc)
    #println("Loss: ", loss)
end
train_model!(model, train_loader, RegressionTask())
acc = calc_accuracy(model, test_loader, RegressionTask())
loss = calc_loss(model, test_loader, RegressionTask())
println("Final accuracy: ", acc)
println("Final loss: ", loss)
punish_model_resetting!(bee1, RegressionTask())


model = build_model(NoTask())
cfg = NoTaskConfig()
train_loader, test_loader, train_data, test_data = create_dataset(NoTask(), cfg)
acc = calc_accuracy(model, test_loader, NoTask())
loss = calc_loss(model, test_loader, NoTask())
println("Initial accuracy: ", acc)
println("Initial loss: ", loss)