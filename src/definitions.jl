"""
The struct @Bee represents one of the neural networks in out simulation of multiple neural networks being trained
    simultaniously whilst interacting with each other. The object holds various other properties apart
    from the neural network.
"""
mutable struct Bee
    id::Integer
    brain::Flux.Chain
    queen_gene::Float64

    function Bee(id::Integer, task::Task, queen_gene::Float64 = 0.0)
        brain_model = build_model(task)
        return new(id, brain_model, queen_gene)
    end
end
    

"""
The struct Hive is the main object on which the simulation is performed. It holds all @Bee objects meaning all 
    neural networks allowing to easily perform all simulation operations on it.
"""
mutable struct Hive
    # all parsed arguments
    n_bees::UInt16
    n_epochs::UInt16
    n_steps_per_epoch::UInt16
    learning_rate::Float16
    punish_rate::Float32
    lambda_train::Float16
    lambda_interact::Float16
    accuracy_sigma::Float16
    task_type::Task
    queen_gene_method::QueenGeneMethod
    bee_list::Vector{Bee}
    epoch_index::UInt16
    # simulation results
    initial_accuracies_list::Vector{Float64}
    queen_genes_history::Matrix{Float64}
    loss_history::Matrix{Float64}
    accuracy_history::Matrix{Float64}
    n_train_history::Matrix{Int}
    n_subdominant_history::Matrix{Int}
    n_dominant_history::Matrix{Int}
    propensity_ratio_history::Vector{Float64} # not recorded yet
end

function Hive(n_bees::UInt16, 
              n_epochs::UInt16,
              n_steps_per_epoch::UInt16,
              learning_rate::Float16,
              punish_rate::Float32,
              lambda_train::Float16,
              lambda_interact::Float16,
              accuracy_sigma::Float16,
              task::Task, 
              queen_gene_method::QueenGeneMethod, 
              bee_list::Vector{Bee})
    # Validate input parameters
    if n_bees != length(bee_list)
        throw(ArgumentError("Number of bees does not match the length of the bee list"))
    end
    if n_epochs < 1
        throw(ArgumentError("Number of epochs must be at least 1"))
    end
    if n_bees < 1
        throw(ArgumentError("Number of bees must be at least 1"))
    end
    if learning_rate <= 0
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
    
    return new(n_bees,
               n_epochs,
               n_steps_per_epoch,
               learning_rate,
               punish_rate,
               lambda_train,
               lambda_interact,
               accuracy_sigma,
               task, 
               queen_gene_method,
               bee_list,
               UInt16(0), #epoch index
               fill(0.0, n_bees), #initial accuracies
               fill(0.0, n_bees, n_epochs), #queen genes history
               fill(0.0, n_bees, n_epochs), #loss history
               fill(-1.0, n_bees, n_epochs), #accuracy history
               fill(0, n_bees, n_epochs), #n_train history
               fill(0, n_bees, n_epochs), #n_subdominant history
               fill(0, n_bees, n_epochs), #n_dominant history
               fill(-1.0, n_epochs) #propensity ratio history
               )
end

function Hive(; 
    n_bees::UInt16 = DEFAULTS[:N_BEES], 
    n_epochs::UInt16 = DEFAULTS[:N_EPOCHS], 
    n_steps_per_epoch::UInt16 = DEFAULTS[:N_STEPS_PER_EPOCH],
    learning_rate::Float16 = DEFAULTS[:LEARNING_RATE],
    punish_rate::Float32 = DEFAULTS[:PUNISH_RATE],
    lambda_train::Float16 = DEFAULTS[:LAMBDA_TRAIN],
    lambda_interact::Float16 = DEFAULTS[:LAMBDA_INTERACT],
    accuracy_sigma::Float16 = DEFAULTS[:ACCURACY_SIGMA],
    task::Task = RegressionTask(),
    queen_gene_method::QueenGeneMethod = QueenGeneFromAccuracy())
    
    # Dynamically generate the bee list
    bee_list = [Bee(UInt16(i), task) for i in 1:n_bees]
    return Hive(n_bees, 
                n_epochs,
                n_steps_per_epoch,
                learning_rate,
                punish_rate,
                lambda_train,
                lambda_interact,
                accuracy_sigma,
                task, 
                queen_gene_method, 
                bee_list)
end

function create_hive(parsed_args::Dict)
    # Extract parsed arguments from the dictionary
    n_bees = get(parsed_args, :n_bees, DEFAULTS[:N_BEES])
    n_epochs = get(parsed_args, :n_epochs, DEFAULTS[:N_EPOCHS])
    n_steps_per_epoch = get(parsed_args, :n_steps_per_epoch, DEFAULTS[:N_STEPS_PER_EPOCH])
    learning_rate = get(parsed_args, :learning_rate, DEFAULTS[:LEARNING_RATE])
    punish_rate = get(parsed_args, :punish_rate, DEFAULTS[:PUNISH_RATE])
    lambda_train = get(parsed_args, :lambda_train, DEFAULTS[:LAMBDA_TRAIN])
    lambda_interact = get(parsed_args, :lambda_interact, DEFAULTS[:LAMBDA_INTERACT])
    accuracy_sigma = get(parsed_args, :accuracy_sigma, DEFAULTS[:ACCURACY_SIGMA])
    task = get(parsed_args, :task, RegressionTask())  # Default to RegressionTask if not provided
    queen_gene_method = get(parsed_args, :queen_gene_method, QueenGeneFromAccuracy())  # Default to QueenGeneFromAccuracy if not provided
    
    # Create bee_list if not provided in the parsed arguments
    bee_list = get(parsed_args, :bee_list, [Bee(UInt16(i), task) for i in 1:n_bees])
    
    # Create and return the Hive object using the constructor
    return Hive(n_bees, 
                n_epochs,
                n_steps_per_epoch,
                learning_rate,
                punish_rate,
                lambda_train,
                lambda_interact,
                accuracy_sigma,
                task, 
                queen_gene_method, 
                bee_list)
end
