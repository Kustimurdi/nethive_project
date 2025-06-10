struct HiveConfig
    dataset_name::String
    parent_dataset_name::String
    n_bees::UInt16
    n_epochs::UInt16
    n_steps_per_epoch::UInt16
    learning_rate::Float32
    punish_rate::Float32
    lambda_train::Float64
    lambda_interact::Float16
    accuracy_sigma::Float16
    random_seed::Int
    task_type::Symbol
    queen_gene_method::Symbol
    task_config::AbstractTaskConfig
    initial_queen_gene::Float64
    save_nn_epochs::Int

    function HiveConfig(dataset_name::String, 
                        parent_dataset_name::String,
                        n_bees::UInt16, 
                        n_epochs::UInt16,
                        n_steps_per_epoch::UInt16,
                        learning_rate::Float32,
                        punish_rate::Float32,
                        lambda_train::Float64,
                        lambda_interact::Float16,
                        accuracy_sigma::Float16,
                        random_seed::Int,
                        task_type::Symbol,
                        queen_gene_method::Symbol,
                        task_config::AbstractTaskConfig,
                        initial_queen_gene::Float64,
                        save_nn_epochs::Int)
        
        # Validate parameters
        if n_bees < 1
            throw(ArgumentError("Number of bees must be at least 1"))
        end
        if n_epochs < 1
            throw(ArgumentError("Number of epochs must be at least 1"))
        end
        if learning_rate <= 0
            println("learning rate: ", learning_rate)
            throw(ArgumentError("Learning rate must be positive")) 
        end
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
                task_config,
                initial_queen_gene,
                save_nn_epochs)
    end

end

function create_hive_config(args)
    dataset_name = string(Dates.format(now(), "DyymmddTHHMMSSss"), "I", rand(1:9, 1)[1])
    task_type = Symbol(args["task_type"])
    queen_gene_method = Symbol(args["queen_gene_method"])
    if task_type == :none && !(queen_gene_method == :incremental)
        throw(ArgumentError("Queen gene method is not supported for no task type"))
    end
    return HiveConfig(
        dataset_name,
        args["parent_dataset_name"],
        UInt16(args["n_bees"]),
        UInt16(args["n_epochs"]),
        UInt16(args["n_steps_per_epoch"]),
        Float32(args["learning_rate"]),
        Float32(args["punish_rate"]),
        Float64(args["lambda_train"]),
        Float16(args["lambda_interact"]),
        Float16(args["accuracy_sigma"]),
        args["random_seed"],
        task_type,
        queen_gene_method,
        TaskConfig(args),
        Float64(args["initial_queen_gene"]),
        Int(args["save_nn_epochs"])
    )
end

"""
The struct @Bee represents one of the neural networks in out simulation of multiple neural networks being trained
    simultaniously whilst interacting with each other. The object holds various other properties apart
    from the neural network.
"""
mutable struct Bee
    id::Integer
    brain::Flux.Chain
    queen_gene::Float64

    function Bee(id::Integer, task::AbstractTask, queen_gene::Float64 = 0.0)
        brain_model = build_model(task)
        return new(id, brain_model, queen_gene)
    end
end
    
mutable struct Hive
    config::HiveConfig
    bee_list::Vector{Bee}
    # simulation results
    queen_genes_history::Matrix{Float64}
    loss_history::Matrix{Float64}
    accuracy_history::Matrix{Float64}
    n_train_history::Matrix{Int}
    n_subdominant_history::Matrix{Int}
    n_dominant_history::Matrix{Int}
    subdominant_rate_history::Matrix{Float64}
    dominant_rate_history::Matrix{Float64}
    propensity_ratio_history::Vector{Float64}
    epoch_index::UInt

    function Hive(config::HiveConfig)
        task = get_task_instance(config.task_type)
        # Create a list of bees dynamically based on the number of bees from the config
        bee_list = [Bee(UInt16(i), task, config.initial_queen_gene) for i in 1:config.n_bees]
        
        # Return a new Hive object using the config object
        return new(config,
                bee_list,
                fill(0.0, config.n_bees, config.n_epochs + 1),  # queen genes history
                fill(0.0, config.n_bees, config.n_epochs),  # loss history
                fill(-1.0, config.n_bees, config.n_epochs + 1),  # accuracy history
                fill(0, config.n_bees, config.n_epochs),  # n_train history
                fill(0, config.n_bees, config.n_epochs),  # n_subdominant history
                fill(0, config.n_bees, config.n_epochs),  # n_dominant history
                fill(-1.0, config.n_bees, config.n_epochs),  # subdominant rate history
                fill(-1.0, config.n_bees, config.n_epochs),  # dominant rate history
                fill(-1.0, config.n_epochs),  # propensity ratio history
                UInt(0)  # epoch index
        )
    end

end

struct HivePaths
    raw_path::String
    raw_net_path::String
    raw_taskdata_path::String
end



function create_hive_paths(config::HiveConfig)
    base_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/"
    dataset_path = joinpath(base_path, config.parent_dataset_name, config.dataset_name)
    #simulation_config_folder_name = "lr_$(config.learning_rate)_pr_$(config.punish_rate)_lt_$(config.lambda_train)_li_$(config.lambda_interact)_as_$(config.accuracy_sigma)"
    #dataset_path = joinpath(base_path, config.parent_dataset_name, simulation_config_folder_name, config.dataset_name)

    return HivePaths(
        dataset_path,
        joinpath(dataset_path, "net"),
        joinpath(dataset_path, "taskdata")
    )
end

