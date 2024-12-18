using Flux
#using Flux.Data: DataLoader
using Flux: DataLoader
using MLDatasets: MNIST
using Statistics: mean
using Random

function build_neural_net(config::Neural_net_config)
    layers = []
    in_size = config.input_size

    for hidden_size in config.hidden_sizes
        push!(layers, Dense(in_size, hidden_size, config.activation))
        in_size = hidden_size
    end

    push!(layers, Dense(in_size, config.output_size))
    push!(layers, softmax)
    return Chain(layers...)
end

function preprocess_greyscale_imagedata(images, labels)
    x = Float32.(reshape(images, size(images, 1) * size(images, 2), size(images, 3)) ./ 255.0)
    y = Flux.onehotbatch(labels .+ 1, 1:10)  # Convert labels to one-hot encoded format
    return x, y
end

function train!(model, dataloader, optimizer, loss_fn, epochs)
    for epoch in 1:epochs
        epoch_loss = 0.0
        for (x_batch, y_batch) in dataloader
            grads = gradient(()->loss_fn(model(x_batch), y_batch), Flux.params(model))
            Flux.Optimise.update!(optimizer, Flux.params(model), grads)
            epoch_loss += loss_fn(model(x_batch), y_batch)
        end
        println("Epoch $epoch, Loss: $(epoch_loss / length(dataloader))")
    end
end

function accuracy(model, dataloader)
    correct = 0
    total = 0
    for (x_batch, y_batch) in dataloader
        preds = onecold(model(x_batch), 1:10)  # Get predicted labels
        truths = onecold(y_batch, 1:10)       # Get true labels
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end

function run()
    config_nn = Neural_net_config(784, [128, 64], 10, relu)
    network = build_neural_net(config_nn)

    Random.seed!(1234)
    train_x, train_y = MNIST(split=:train)[:]
    test_x, test_y = MNIST(split=:test)[:]
    toy_network = Chain(Dense(784,10), softmax)

    x_train, y_train = preprocess_greyscale_imagedata(train_x, train_y)
    x_test, y_test = preprocess_greyscale_imagedata(test_x, test_y)
    x_example = x_train[:,1]
    y_example = y_train[:,1]

    println(toy_network(x_example))
    loss(x, y) = Flux.crossentropy(toy_network(x), y, dims = 10)
    println(loss(x_example, y_example))
    #=    
    train_loader = DataLoader((x_train, y_train), batchsize=128, shuffle=true)
    test_loader = DataLoader((x_test, y_test), batchsize=128)

    loss_fn(y_pred, y_true) = mean(crossentropy(y_pred, y_true))  # Cross-entropy loss
    optimizer = Flux.setup(Adam(), network)  # Stochastic Gradient Descent with learning rate 0.01

    train!(network, train_loader, optimizer, loss_fn, 10)

    println("Test Accuracy: $(accuracy(model, test_loader) * 100)%")
    =#
end