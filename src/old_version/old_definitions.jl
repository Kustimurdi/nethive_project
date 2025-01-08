
mutable struct Config_Fnn
    input_size::Int
    hidden_sizes::Vector{Int}
    output_size::Int
    activation::Function
end

mutable struct Hive
    n_neural_nets::Int
end
