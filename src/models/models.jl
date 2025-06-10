function build_model(::RegressionTask)
    return build_model_sin_leaky()
end

function build_model(::LinearRegressionTask)
    return build_model_linear()
end

function build_model(task::ClassificationTask)
    return build_classification_model(input_size=task.input_size, output_size=task.output_size)
end

function build_model(::NoTask)
    return Flux.Chain()  # Empty model
end


"""
the function @build_brain creates the neural networks for the @Bee objects (the struct objects that hold the 
    neural networks and various other pieces of important information concerning the simulation) of the @Hive
    At the moment the used NN architecture is a CNN with two CNN layers and two FNN layers.
"""

function build_model_sin()
    return Chain(
        Dense(1, 16, relu),
        Dense(16, 16, relu),
        Dense(16, 1)
    )
end

function build_model_sin_leaky()
    return Chain(
        Dense(1, 16, leakyrelu; init=Flux.glorot_normal),
        Dense(16, 16, leakyrelu; init=Flux.glorot_normal),
        Dense(16, 1; init=Flux.glorot_normal)
    )
end

function build_model_linear()
    return Chain(Dense(1, 1))  # Simple linear regression: y = Wx + b
end

function build_classification_model(; input_size::AbstractVector{<:Integer}, output_size::UInt16)

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


function build_model_5(; input_size::AbstractVector{<:Integer}, output_size::UInt16)

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