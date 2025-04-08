"""
The struct @Bee represents one of the neural networks in out simulation of multiple neural networks being trained
    simultaniously whilst interacting with each other. The object holds various other properties apart
    from the neural network.
"""
struct HiveConfig
    dataset_name::String
    parent_dataset_name::String
    n_bees::UInt16
    n_epochs::UInt16
    n_steps_per_epoch::UInt16
    learning_rate::Float16
    punish_rate::Float32
    lambda_train::Float16
    lambda_interact::Float16
    accuracy_sigma::Float16
    random_seed::Int
    task_type::Symbol
    queen_gene_method::Symbol
    task_config::AbstractTaskConfig
end

mutable struct Bee
    id::Integer
    brain::Flux.Chain
    queen_gene::Float64

    function Bee(id::Integer, task::Task, queen_gene::Float64 = 0.0)
        brain_model = build_model(task)
        return new(id, brain_model, queen_gene)
    end
end
    
mutable struct Hive
    config::HiveConfig
    bee_list::Vector{Bee}
    # simulation results
    initial_accuracies_list::Vector{Float64}
    queen_genes_history::Matrix{Float64}
    loss_history::Matrix{Float64}
    accuracy_history::Matrix{Float64}
    n_train_history::Matrix{Int}
    n_subdominant_history::Matrix{Int}
    n_dominant_history::Matrix{Int}
    propensity_ratio_history::Vector{Float64}
end

struct HivePaths
    raw_path::String
    raw_net_path::String
    raw_taskdata_path::String
end

function HiveConfig(dataset_name::String, 
                    parent_dataset_name::String,
                    n_bees::UInt16, 
                    n_epochs::UInt16,
                    n_steps_per_epoch::UInt16,
                    learning_rate::Float16,
                    punish_rate::Float32,
                    lambda_train::Float16,
                    lambda_interact::Float16,
                    accuracy_sigma::Float16,
                    random_seed::Int,
                    task_type::Symbol,
                    queen_gene_method::Symbol,
                    task_config::AbstractTaskConfig)
    
    # Validate parameters
    if n_bees < 1
        throw(ArgumentError("Number of bees must be at least 1"))
    end
    if n_epochs < 1
        throw(ArgumentError("Number of epochs must be at least 1"))
    end
    if learning_rate <= 0
        throw(ArgumentError("Learning rate must be positive")) end
    if punish_rate <= 0
        throw(ArgumentError("Punish rate must be positive"))
    end
    if lambda_train <= 0 || lambda_interact <= 0
        throw(ArgumentError("Lambda values must be positive"))
    end
    if accuracy_sigma <= 0
        throw(ArgumentError("Accuracy sigma must be positive"))
    end

    return new(dataset_name, 
               parent_dataset_name, 
               n_bees, 
               n_epochs, 
               n_steps_per_epoch,
               learning_rate, 
               punish_rate, 
               lambda_train, 
               lambda_interact, 
               accuracy_sigma,
               random_seed,
               task_type,
               queen_gene_method,
               task_config)
end

#hier muss noch angepasst werden, was die default werte sind, und ob nun der abstract type oder symbole benutzt werden
function create_hive_config(args)
    dataset_name = string(Dates.format(now(), "DyymmddTHHMMSSss"), "I", rand(1:9, 1)[1])
    return HiveConfig(
        dataset_name,
        args["parent_dataset_name"],
        UInt16(args["n_bees"]),
        UInt16(args["n_epochs"]),
        UInt16(args["n_steps_per_epoch"]),
        Float16(args["learning_rate"]),
        Float32(args["punish_rate"]),
        Float16(args["lambda_train"]),
        Float16(args["lambda_interact"]),
        Float16(args["accuracy_sigma"]),
        args["random_seed"],
        Symbol(args["task_type"]),
        Symbol(args["queen_gene_method"]),
        TaskConfig(args)
    )
end


function Hive(config::HiveConfig)
    # Create a list of bees dynamically based on the number of bees from the config
    bee_list = [Bee(UInt16(i), task) for i in 1:config.n_bees]
    
    # Return a new Hive object using the config object
    return new(config,
               bee_list,
               fill(0.0, config.n_bees),  # initial accuracies
               fill(0.0, config.n_bees, config.n_epochs),  # queen genes history
               fill(0.0, config.n_bees, config.n_epochs),  # loss history
               fill(-1.0, config.n_bees, config.n_epochs),  # accuracy history
               fill(0, config.n_bees, config.n_epochs),  # n_train history
               fill(0, config.n_bees, config.n_epochs),  # n_subdominant history
               fill(0, config.n_bees, config.n_epochs),  # n_dominant history
               fill(-1.0, config.n_epochs)  # propensity ratio history
    )
end

function create_hive_paths(config::HiveConfig)
    base_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/"
    dataset_path = joinpath(base_path, config.parent_dataset_name, config.dataset_name)

    return HivePaths(
        dataset_path,
        joinpath(dataset_path, "net"),
        joinpath(dataset_path, "taskdata")
    )
end

