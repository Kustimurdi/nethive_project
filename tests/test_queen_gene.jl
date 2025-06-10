include("../src/tasks/task_types.jl")
include("../src/tasks/task_training.jl")

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

task = get_task_instance(args["task_type"])
taskconfig = TaskConfig(args)
trainloader, testloader, train_data, test_data = create_dataset(task, taskconfig)
initial_queen_gene = 0.3

@testset "QueenGeneFromAccuracy" begin
    # Create mock data for testing
    
    bee = Bee(1, task, initial_queen_gene)  # Example bee with initial queen gene value

    # Compute queen gene using QueenGeneFromAccuracy
    method = QueenGeneFromAccuracy()
    queen_gene = compute_queen_gene(bee, testloader, task, method)

    @test queen_gene == calc_accuracy(bee.brain, testloader, task)  # Assuming calc_accuracy is available
    train_model!(bee.brain, trainloader, task, learning_rate=0.01)  # Train the model

    queen_gene2 = compute_queen_gene(bee, testloader, task, method)
    @test queen_gene2 == calc_accuracy(bee.brain, testloader, task)  # Check if the queen gene is updated
    @test queen_gene2 != queen_gene  # Ensure the queen gene has changed after training
    @test queen_gene2 > queen_gene  # Ensure the queen gene has increased after training
end

# Test QueenGeneFromLoss computation
@testset "QueenGeneFromLoss" begin
    # Create mock data for testing
    bee = Bee(1, task, initial_queen_gene)  # Example bee with initial queen gene value

    # Compute queen gene using QueenGeneFromLoss
    method = QueenGeneFromLoss()
    result = compute_queen_gene(bee, testloader, task, method)

    # Check if result matches the expected formula: (1 / (loss + 1))
    expected_result = 1 / (calc_loss(bee.brain, testloader, task) + 1)  # Assuming calc_loss is available
    #@test result == expected_result
    @test isapprox(result, expected_result)
    println("difference between result and expected_result: ", abs(result - expected_result))

    train_model!(bee.brain, trainloader, task, learning_rate=0.01)  # Train the model
    result2 = compute_queen_gene(bee, testloader, task, method)
    expected_result2 = 1 / (calc_loss(bee.brain, testloader, task) + 1)  # Assuming calc_loss is available
    #@test result == expected_result
    @test isapprox(result2, expected_result2)  # Check if the queen gene is updated
    println("difference between result2 and expected_result: ", abs(result2 - expected_result))
    @test result2 != result  # Ensure the queen gene has changed after training
    @test result2 > result  # Ensure the queen gene has decreased after training
end

# Test QueenGeneIncremental computation
@testset "QueenGeneIncremental" begin
    # Create mock data for testing
    task = NoTask()
    bee = Bee(1, task, 0.3)  # Example bee with initial queen gene value
    increment_value = 0.2
    method = QueenGeneIncremental(increment_value)

    # Compute queen gene using QueenGeneIncremental
    result = compute_queen_gene(bee, method, nothing, nothing)

    # Check if result matches the expected increment value
    expected_result = bee.queen_gene + increment_value
    @test result == expected_result
end

# Test handling of invalid QueenGeneMethod
@testset "Invalid QueenGeneMethod" begin
    # Create a dummy method that doesn't exist
    invalid_method = "InvalidMethod"  # This should cause an error
    @test_throws ArgumentError begin
        compute_queen_gene(Bee(1.0, 5.0), nothing, nothing, invalid_method)
    end
end