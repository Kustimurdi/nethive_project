
mutable struct Neural_net_config
    input_size::Int
    hidden_sizes::Vector{Int}
    output_size::Int
    activation::Function
end

mutable struct hive
    n_neural_nets::Int
end
