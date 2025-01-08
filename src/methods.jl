using Flux, MLDatasets

function build_brain(input_size::Array{UInt16}, output_size::UInt16)
    layers = []
    push!(layers, Flux.Conv((5,5), input_size[3] => 10, pad=2, relu))
    push!(layers, Flux.MaxPool((2,2), stride=2))

    push!(layers, Flux.Conv((5,5), 10 => 20, pad=2, relu))
    push!(layers, Flux.MaxPool((2,2), stride=2))
    push!(layers, BatchNorm(10)) 

    push!(layers, Flux.flatten)
    input_size_dense = Int64((input_size[1]/4) * (input_size[2]/4) * 20)
    println("bis")
    println(input_size_dense)
    push!(layers, Flux.Dense(input_size_dense, 128, relu))
    push!(layers, Flux.Dense(128, output_size))
    push!(layers, Flux.softmax)
    return Chain(layers...)
end

function prepare_MNIST(normalize::Bool=false)
    x_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
    x_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]
    y_train = onehotbatch(y_train_raw, 0:9)
    y_test = onehotbatch(y_test_raw, 0:9)
    if normalize == true
        x_train_raw = x_test_raw ./ 255.0
        x_test_raw = x_test_raw ./ 255.0
    end
    data = [x_train_raw, y_train, x_test_raw, y_test]
    return data
end


function run(h::Hive, n_epochs::UInt8)
    learning_rate = Float16(0.01)
    opt = Flux.Adam(learning_rate) 
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
    data = prepare_MNIST()
    
    train_loader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
    test_loader = Flux.DataLoader((data[3], data[4]), batchsize=128)

    for epoch = 1:n_epochs
        for bee in h.bee_list
            train!(bee)
        end
    end
    return 0
end




"""
------------------------------------------
Testing
------------------------------------------
"""
a = [10, 10, UInt8(1)]
println(type(a))

h = Hive(UInt16(5))

run(h, UInt8(10))

build_brain([UInt16(28), UInt16(28), UInt16(1)], UInt16(10))