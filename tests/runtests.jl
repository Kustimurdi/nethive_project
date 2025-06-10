# Activate project environment
import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")
using Revise
#Pkg.instantiate()

# === Load all required source files in dependency-safe order ===

# 1. Dependencies and general helpers
include("../src/dependencies.jl")
include("../src/helper.jl")  # Uses the packages defined above

# 2. Config and argument parsing
include("../src/config/defaults.jl")
include("../src/config/arg_table.jl")
include("../src/config/parse_args.jl")

# 3. Task system
include("../src/tasks/task_types.jl")     # AbstractTaskConfig, Task, etc.
include("../src/tasks/task_utils.jl")     # TaskConfig, accuracy, loss, etc.

# 4. Models
include("../src/models/models.jl")        # build_model, etc.

# 5. Core simulation logic
include("../src/core/definitions.jl")     # Bee, Hive, HiveConfig, etc.
include("../src/core/queen_gene.jl")      # QueenGeneMethod types and compute logic
include("../src/core/methods.jl")         # punish_model, gillespie_simulation, etc.
include("../src/tasks/task_training.jl")  # Training-related functions

# 6. Simulation entry point
include("../src/simulation.jl")

# === Include test files ===

# include("test_parse_args.jl")
# include("test_helper.jl")
# include("test_tasks.jl")
# include("test_models.jl")
# include("test_definitions.jl")
# include("test_methods.jl")
# include("test_simulation.jl")


const custom = Dict(
    :parent_dataset_name => "testing",
    :task_type => :regression,
    :queen_gene_method => :accuracy,
    :n_bees => UInt16(3),
    :n_epochs => UInt16(1500),
    :n_steps_per_epoch => 1,
    :learning_rate => Float16(0.00001),
    :punish_rate => Float32(0.00001),
    :lambda_train => Float16(0.1),
    :lambda_interact => Float16(5),
    :accuracy_sigma => Float16(0.1),
    :random_seed => 2,
    :trainset_size => 10000,
    :testset_size => 1000,
    :initial_queen_gene => Float64(0.0),

    # regression defaults
    :regression_n_peaks => 5,
    :regression_which_peak => 1
)



s = create_arg_parse_settings(custom)
parsed_args = parse_args(s)

hive = run_simulation(parsed_args; save_data=false, verbose=true, seed=parsed_args["random_seed"])

include("../analysis/analysis_jl/extra/methods_repl.jl")
queen_fig = plot_hive_history(hive.queen_genes_history, title="queen gene History", xlabel="Epochs", ylabel="queen gene")
acc_fig = plot_hive_history(hive.accuracy_history, title="Accuracy History", xlabel="Epochs", ylabel="Accuracy")
lds = create_linear_dataset(10000)
sin_ds = create_sin_dataset(parsed_args["regression_n_peaks"], parsed_args["regression_which_peak"], 10000)
plot_dataset(lds)
plot_hive_predictions(hive, sin_ds[1])


plot_dataset(lds)
sum(hive.n_dominant_history, dims=2)
sum(hive.n_train_history, dims=2)
hive.accuracy_history


#dom_fig = plot_hive_history(hive.n_dominant_history, title="Dominance History", xlabel="Epochs", ylabel="Dominance", which_bee=1)
propensity_fig = plot_dataset(hive.propensity_ratio_history, title="Propensity Ratio History", xlabel="Epochs", ylabel="Propensity Ratio")
train_fig = plot_hive_history(hive.n_train_history, title="Train History", xlabel="Epochs", ylabel="Train")
train_sum = vec(sum(hive.n_train_history, dims=1))
train_sum_fig = plot_dataset(train_sum, title="Train History", xlabel="Epochs", ylabel="Train")
interaction_sum = vec(sum(hive.n_dominant_history, dims=1)) #+ vec(sum(hive.n_subdominant_history, dims=1))
interaction_sum_fig = plot_dataset(interaction_sum, title="Interaction History", xlabel="Epochs", ylabel="Interaction")


qg_dummy = [10, 1, 0.1]
lambda_interact_dummy = 5
mat = compute_K_matrix(queen_genes_list = qg_dummy, lambda_interact=lambda_interact_dummy)
"""