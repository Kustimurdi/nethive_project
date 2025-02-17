"""
The file @definitions.jl holds the defintions of all of the objects relevant for the simulation (the objects on
which the simulation will operate) and all the necessary functions for the structs
"""
#using Flux 

"""
The struct @Bee represents one of the neural networks in out simulation of multiple neural networks being trained
    simultaniously whilst interacting with each other. The object holds various other properties apart
    from the neural network.
"""
mutable struct Bee
    id::Integer
    brain::Flux.Chain
    params_history::Dict{Int, Any}
    function Bee(id::Integer, 
                 brain_constructor::Function = build_brain,
                 input_size::AbstractVector{<:Integer} = DEFAULTS[:INPUT_SIZE],
                 output_size::UInt16 = DEFAULTS[:OUTPUT_SIZE])
        
        brain_model = brain_constructor(input_size, output_size)
        params_dict = Dict{Int, Any}()
        params_dict[0] = deepcopy(Flux.params(brain_model))

        new(id,
        brain_model,
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
    loss_history::Matrix{Float64}
    accuracy_history::Matrix{Float64}
    interaction_partner_history::Matrix{Int}
    interaction_results_history::Matrix{Int}
    epoch_index::UInt
    function Hive(n_bees::UInt16 = DEFAULTS[:N_BEES], 
                  n_epochs::UInt16 = DEFAULTS[:N_EPOCHS], 
                  #brain_constructor::Function = build_brain(input_size::AbstractVector{<:Integer}, output_size::UInt16)) #can use const global for default values later
                  input_size::AbstractVector{<:Integer} = DEFAULTS[:INPUT_SIZE],
                  output_size::UInt16 = DEFAULTS[:OUTPUT_SIZE],
                  brain_constructor::Function = build_brain)

        bee_list = Vector{Bee}(undef, n_bees)
        for i = 1:n_bees
            bee_list[i] = Bee(UInt16(i), brain_constructor, input_size, output_size)
        end

        losses = fill(-1, n_bees, n_epochs)
        accuracies = fill(-1, n_bees, n_epochs + 1) #the matrix has one additional column for the initial accuracies
        interaction_partners = fill(-1, n_bees, n_epochs)
        interaction_results = fill(-1, n_bees, n_epochs)

        new(n_bees::UInt16,
        bee_list,
        losses,
        accuracies,
        interaction_partners,
        interaction_results,
        UInt(0))
    end
end

"""
the function @build_brain creates the neural networks for the @Bee objects (the struct objects that hold the 
    neural networks and various other pieces of important information concerning the simulation) of the @Hive
    At the moment the used NN architecture is a CNN with two CNN layers and two FNN layers.
"""
function build_brain(input_size::AbstractVector{<:Integer}, output_size::UInt16)
    layers = []
    push!(layers, Flux.Conv((5,5), input_size[3] => 10, pad=2, relu))
    push!(layers, Flux.MaxPool((2,2), stride=2))

    push!(layers, Flux.Conv((5,5), 10 => 20, pad=2, relu))
    push!(layers, Flux.MaxPool((2,2), stride=2))
    push!(layers, BatchNorm(20)) 

    push!(layers, Flux.flatten)
    input_size_dense = Int64((input_size[1]/4) * (input_size[2]/4) * 20)
    push!(layers, Flux.Dense(input_size_dense, 128, relu))
    push!(layers, Flux.Dense(128, output_size))
    push!(layers, Flux.softmax)
    return Chain(layers...)
end






