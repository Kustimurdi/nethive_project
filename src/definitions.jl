"""
The file @definitions.jl holds the defintions of all of the objects relevant for the simulation (the objects on
which the simulation will operate) and all the necessary functions for the structs
"""

module Definitions

#include("config/defaults.jl")
using .Defaults
using Flux 

export Hive, Bee

"""
The struct @Bee represents one of the neural networks in out simulation of multiple neural networks being trained
    simultaniously whilst interacting with each other. The object holds various other properties apart
    from the neural network.
"""
mutable struct Bee
    id::Integer
    brain::Flux.Chain
    loss_history::Array{Float32}
    accuracy_history::Array{Float32}
    params_history::Dict{Int, Any}
    function Bee(id::Integer, 
                 n_epochs::UInt16 = Defaults.DEFAULTS[:N_EPOCHS], 
                 brain_constructor::Function = build_brain,
                 input_size::AbstractVector{<:Integer} = Defaults.DEFAULTS[:INPUT_SIZE],
                 output_size::UInt16 = Defaults.DEFAULTS[:OUTPUT_SIZE])
        
        brain_model = brain_constructor(input_size, output_size)
        params_dict = Dict{Int, Any}()
        params_dict[0] = deepcopy(Flux.params(brain_model))

        new(id,
        brain_model,
        zeros(Float32, n_epochs),
        zeros(Float32, n_epochs),
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
    function Hive(n_bees::UInt16 = Defaults.DEFAULTS[:N_BEES], 
                  n_epochs::UInt16 = Defaults.DEFAULTS[:N_EPOCHS], 
                  #brain_constructor::Function = build_brain(input_size::AbstractVector{<:Integer}, output_size::UInt16)) #can use const global for default values later
                  brain_constructor::Function = build_brain,
                  input_size::AbstractVector{<:Integer} = Defaults.DEFAULTS[:INPUT_SIZE],
                  output_size::UInt16 = Defaults.DEFAULTS[:OUTPUT_SIZE])

        bee_list = Vector{Bee}(undef, n_bees)
        for i = 1:n_bees
            bee_list[i] = Bee(UInt16(i), n_epochs::UInt16, brain_constructor, input_size, output_size)
        end

        new(n_bees::UInt16,
        bee_list)
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

end
"""End of the module"""





