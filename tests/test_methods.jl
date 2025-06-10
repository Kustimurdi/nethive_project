using Test
using Random

# Import necessary modules
include("../src/core/methods.jl")
include("../src/core/definitions.jl")
include("../src/core/queen_gene.jl")  # Assuming you've written tests for these methods too

# Test compute_K_matrix
@testset "compute_K_matrix" begin
    queen_genes_list = [0.1, 0.3]
    lambda_interact = 1000000
    K_matrix = compute_K_matrix(queen_genes_list=queen_genes_list, lambda_interact=lambda_interact)
    #@test size(K_matrix) == (I, 3)  # Check matrix size
    @test K_matrix[1, 1] == 0.0    # Check diagonal elements are 0
    println("K_matrix: ", K_matrix)
end

# Test K_func
@testset "K_func" begin
    r_i = 0.5
    r_j = 0.2
    lambda = 100000000
    result = K_func(r_i, r_j, lambda)
    #@test result ≈ 0.5 * 0.6 * Flux.sigmoid(-lambda * (r_i - r_j))  # Check the formula
    println("K_func result: ", result)
end

# Test choose_interaction function
@testset "choose_interaction" begin
    Random.seed!(43)  # For reproducibility
    settings = create_arg_parse_settings()    
    args = cparse_args(settings)
    config = create_hive_config(args)
    hive = Hive(config)
    println("hive n bees: ", hive.config.n_bees)

    K_matrix = compute_K_matrix(queen_genes_list=[bee.queen_gene for bee in hive.bee_list], lambda_interact=0.5)
    println("K_matrix: ", K_matrix)
    println("K_matrix_sum: ", sum(K_matrix))
    chosen_bees_id = choose_interaction(hive.config.n_bees, K_matrix)
    println("chosen_bees_id: ", chosen_bees_id)
    @test chosen_bees_id isa Tuple
    @test length(chosen_bees_id) == 2  # Check if two bees are chosen
    @test all(1 <= id <= hive.config.n_bees for id in chosen_bees_id)  # Check if chosen IDs are valid
    @test chosen_bees_id[1] != chosen_bees_id[2]  # Ensure two different bees are chosen
end

# Test Gillespie simulation behavior
@testset "Gillespie simulation" begin

    Random.seed!(43)  # For reproducibility
    settings = create_arg_parse_settings(DEFAULTS)    
    args = cparse_args(settings)
    config = create_hive_config(args)
    hive = Hive(config)
    h_paths = create_hive_paths(config)
    println(hive.epoch_index)
    println("h paths: ", h_paths)

    
    """
    # Mock functions like create_dataset, save_taskdata, etc., or create minimal mock versions
    function create_dataset(task_type, task_config)
        return ([], [], [], [])  # Return empty datasets for testing
    end

    function save_taskdata(path, train_data, test_data)
        return nothing
    end
    """

    # Mock the Gillespie simulation
    gillespie_simulation!(hive, h_paths)

    # Test after Gillespie simulation
    @test hive.epoch_index > 0  # Ensure some epochs were completed
    #@test all(isapprox(hive.queen_genes_history[:, epoch], hive.queen_genes_history[:, epoch-1], atol=1e-6) for epoch in 2:hive.epoch_index)
end


Random.seed!(43)  # For reproducibility
const custom = Dict(
    :parent_dataset_name => "testing",
    :task_type => :regression,
    :queen_gene_method => :accuracy,
    :n_bees => UInt16(4),
    :n_epochs => UInt16(3000),
    :n_steps_per_epoch => 1,
    :learning_rate => Float16(0.00003),
    :punish_rate => Float32(0.0000001),
    :lambda_train => Float16(0.005),
    :lambda_interact => Float16(5),
    :accuracy_sigma => Float16(0.1),
    :random_seed => 1,
    :trainset_size => 10000,
    :testset_size => 1000,
    :initial_queen_gene => Float64(0.5),

    # regression defaults
    :regression_n_peaks => 5,
    :regression_which_peak => 1
)

settings = create_arg_parse_settings(custom)    
args = cparse_args(settings)
config = create_hive_config(args)
hive = Hive(config)
h_paths = create_hive_paths(config)
traind, testd, trainl, testl = gillespie_simulation!(hive, h_paths)
acc_fig = plot_hive_history(hive.accuracy_history, title="Accuracy History", xlabel="Epochs", ylabel="Accuracy")
acc_fig
fig_path = joinpath("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_images", config.parent_dataset_name, config.dataset_name)
home_path = joinpath("expanduser("~"), /nethive_images", config.parent_dataset_name, config.dataset_name)
mkpath(fig_path)
mkpath(home_path)
save(joinpath(home_path, "accuracy_history.svg"), acc_fig)
save(joinpath(fig_path, "accuracy_history.svg"), acc_fig)
save_simulation_params(config, h_paths.raw_path)
plot_hive_history(hive.queen_genes_history)
plot_hive_history(hive.loss_history)
plot_hive_history(hive.n_train_history)
plot_hive_history(hive.n_dominant_history)
plot_hive_history(hive.n_subdominant_history)
plot_dataset(hive.propensity_ratio_history)


plot_dataset(traind)
plot_bee_prediction(hive.bee_list[1].brain, traind[1])
traind
hive.bee_list[1].brain(traind[1])

x = LinRange(-50, 50, 100)
l = 0.05
y = Flux.sigmoid.(l*x)
plot_dataset([x, y])