"""
The file @definitions.jl holds the defintions of all of the objects relevant for the simulation (the objects on
which the simulation will operate) and all the necessary functions for the structs
"""

"""
The struct @Bee represents one of the neural networks in out simulation of multiple neural networks being trained
    simultaniously whilst interacting with each other. The object holds various other properties apart
    from the neural network.
"""
mutable struct Bee
    id::Integer
    brain::Flux.Chain
    current_accuracy::Float32
    params_history::Dict{Int, Any}
    function Bee(id::Integer, 
                 brain_constructor::Function = build_brain,
                 input_size::AbstractVector{<:Integer} = DEFAULTS[:INPUT_SIZE],
                 output_size::UInt16 = DEFAULTS[:OUTPUT_SIZE])
        
        brain_model = brain_constructor(input_size=input_size, output_size=output_size)
        params_dict = Dict{Int, Any}()
        params_dict[0] = deepcopy(Flux.params(brain_model))

        new(id,
        brain_model,
        Float32(0),
        params_dict)
    end
end

"""
The struct Hive is the main object on which the simulation is performed. It holds all @Bee objects meaning all 
    neural networks allowing to easily perform all simulation operations on it.
"""
mutable struct Hive
    n_bees::UInt16
    bee_list::Vector{Bee}
    current_accuracies_list::Vector{Float64}
    initial_accuracies_list::Vector{Float64}
    loss_history::Matrix{Float64}
    accuracy_history::Matrix{Float64}
    n_subdom_interactions_history::Matrix{Int}
    n_dom_interactions_history::Matrix{Int}
    propensity_ratio_history::Vector{Float64}
    epoch_index::UInt
    function Hive(n_bees::UInt16 = DEFAULTS[:N_BEES], 
                  n_epochs::UInt16 = DEFAULTS[:N_EPOCHS]; 
                  input_size::AbstractVector{<:Integer} = DEFAULTS[:INPUT_SIZE],
                  output_size::UInt16 = DEFAULTS[:OUTPUT_SIZE],
                  brain_constructor::Function = build_model_sin_adaptive)

        bee_list = Vector{Bee}(undef, n_bees)
        for i = 1:n_bees
            bee_list[i] = Bee(UInt16(i), brain_constructor, input_size, output_size)
        end

        current_accuracies = fill(-1, n_bees)
        initial_accuracies = fill(-1, n_bees)
        losses = fill(0, n_bees, n_epochs)
        accuracies = fill(-1, n_bees, n_epochs)
        n_subdom_interactions = fill(0, n_bees, n_epochs)
        n_dom_interactions = fill(0, n_bees, n_epochs)
        propensity_ratios = fill(-1, n_epochs)

        new(n_bees::UInt16,
        bee_list,
        current_accuracies,
        initial_accuracies,
        losses,
        accuracies,
        n_subdom_interactions,
        n_dom_interactions,
        propensity_ratios,
        UInt(0))
    end
end

"""
the function @build_brain creates the neural networks for the @Bee objects (the struct objects that hold the 
    neural networks and various other pieces of important information concerning the simulation) of the @Hive
    At the moment the used NN architecture is a CNN with two CNN layers and two FNN layers.
"""

function build_model_4(; input_size::AbstractVector{<:Integer}=DEFAULTS[:INPUT_SIZE], output_size::UInt16=DEFAULTS[:OUTPUT_SIZE])

    return Chain(
        # Convolutional Block 1
        Conv((3, 3), input_size[3] => 16, relu, pad=1), BatchNorm(16), Dropout(0.2),
        Conv((3, 3), 16 => 16, relu, pad=1), BatchNorm(16), MaxPool((2, 2)),

        # Convolutional Block 2
        Conv((3, 3), 16 => 16, relu, pad=1), BatchNorm(16), Dropout(0.3),
        Conv((3, 3), 16 => 32, relu, pad=1), BatchNorm(32), MaxPool((2, 2)),

        # Convolutional Block 3
        Conv((3, 3), 32 => 32, relu, pad=1), BatchNorm(32), Dropout(0.4),
        Conv((3, 3), 32 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),

        # Flatten and Fully Connected Layers
        Flux.flatten,
        Dense(64 * Int64(input_size[1]/8) * Int64(input_size[2]/8), 256, relu), Dropout(0.5),
        Dense(256, 128, relu), Dropout(0.5),
        Dense(128, output_size)  # CIFAR-10 has 10 classes
    )
end


function build_model_5(; input_size::AbstractVector{<:Integer}=DEFAULTS[:INPUT_SIZE], output_size::UInt16=DEFAULTS[:OUTPUT_SIZE])

    return Chain(
        # Convolutional Block 1
        Conv((3, 3), input_size[3] => 32, relu, pad=1), BatchNorm(32), Dropout(0.2),
        Conv((3, 3), 32 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),

        # Convolutional Block 2
        Conv((3, 3), 64 => 96, relu, pad=1), BatchNorm(96), Dropout(0.3),
        Conv((3, 3), 96 => 128, relu, pad=1), BatchNorm(128), MaxPool((2, 2)),

        # Convolutional Block 3
        Conv((3, 3), 128 => 192, relu, pad=1), BatchNorm(192), Dropout(0.4),
        Conv((3, 3), 192 => 256, relu, pad=1), BatchNorm(256), MaxPool((2, 2)),

        # Flatten and Fully Connected Layers
        Flux.flatten,
        Dense(256 * Int64(input_size[1]/8) * Int64(input_size[2]/8), 256, relu), Dropout(0.5),
        #Dense(256, 128, relu), Dropout(0.5),
        Dense(256, output_size)  # CIFAR-10 has 10 classes
    )
end

function build_model_sin()
    return Chain(
        Dense(1, 16, relu),
        Dense(16, 16, relu),
        Dense(16, 1)
    )
end


mutable struct Bee_fixed_dims
    id::Integer
    brain::Flux.Chain
    current_accuracy::Float32
    params_history::Dict{Int, Any}
    function Bee_fixed_dims(id::Integer, brain_constructor::Function = build_model_sin)
        
        brain_model = brain_constructor()
        params_dict = Dict{Int, Any}()
        params_dict[0] = deepcopy(Flux.params(brain_model))

        new(id,
        brain_model,
        Float32(0),
        params_dict)
    end
end

mutable struct Hive_fixed_dims
    n_bees::UInt16
    bee_list::Vector{Bee_fixed_dims}
    current_accuracies_list::Vector{Float64}
    initial_accuracies_list::Vector{Float64}
    loss_history::Matrix{Float64}
    accuracy_history::Matrix{Float64}
    n_subdom_interactions_history::Matrix{Int}
    n_dom_interactions_history::Matrix{Int}
    propensity_ratio_history::Vector{Float64}
    epoch_index::UInt
    function Hive_fixed_dims(n_bees::UInt16 = DEFAULTS[:N_BEES], 
                  n_epochs::UInt16 = DEFAULTS[:N_EPOCHS]; 
                  brain_constructor::Function = build_model_sin)

        bee_list = Vector{Bee_fixed_dims}(undef, n_bees)
        for i = 1:n_bees
            bee_list[i] = Bee_fixed_dims(UInt16(i), brain_constructor)
        end

        current_accuracies = fill(-1, n_bees)
        initial_accuracies = fill(-1, n_bees)
        losses = fill(0, n_bees, n_epochs)
        accuracies = fill(-1, n_bees, n_epochs)
        n_subdom_interactions = fill(0, n_bees, n_epochs)
        n_dom_interactions = fill(0, n_bees, n_epochs)
        propensity_ratios = fill(-1, n_epochs)

        new(n_bees::UInt16,
        bee_list,
        current_accuracies,
        initial_accuracies,
        losses,
        accuracies,
        n_subdom_interactions,
        n_dom_interactions,
        propensity_ratios,
        UInt(0))
    end
end
