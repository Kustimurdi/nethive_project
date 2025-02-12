using Flux
using Flux: crossentropy, onecold, onehotbatch, params, train!, DataLoader
using MLDatasets: MNIST
using Statistics: mean
using Random
using Images

struct Config_Fnn
    input_size::UInt16
    hidden_size::Array{UInt16}
    ouput_size::UInt16
    activation::Function
end

function build_fnn(config::Config_Fnn)
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
    x = Float32.(reshape(images, size(images, 1) * size(images, 2), size(images, 3)))
    #x = Float32.(reshape(images, size(images, 1) * size(images, 2), size(images, 3)) ./ 255.0)
    y = Flux.onehotbatch(labels, 0:9)
    #y = Flux.onehotbatch(labels .+ 1, 1:10)  # Convert labels to one-hot encoded format
    return x, y
end

function custom_train!(model, dataloader, optimizer, loss_fn, epochs)
    loss_history = []
    for epoch in 1:epochs
        epoch_loss = 0.0
        for (x_batch, y_batch) in dataloader
            println(size(x_batch))
            println(size(y_batch))
            model(x_batch)
            loss_fn(model(x_batch), y_batch)
            grads = gradient(()->loss_fn(model(x_batch), y_batch), Flux.params(model))
            Flux.Optimise.update!(optimizer, Flux.params(model), grads)
            epoch_loss += loss_fn(model(x_batch), y_batch)
        end
        push!(loss_history, epoch_loss)
        println("Epoch $epoch, Loss: $(epoch_loss / length(dataloader))")
    end
    return loss_history
end

function basic_train!(model, optimizer, loss_fn, epochs, X_train, y_train)
    loss_history = []

    for epoch in 1:epochs
        # train model
        Flux.train!(loss_fn, Flux.params(model), [(X_train, y_train)], optimizer)
        # print report
        train_loss = loss_fn(X_train, y_train)
        push!(loss_history, train_loss)
        println("Epoch = $epoch : Training Loss = $train_loss")
    end
    return loss_history
end


function accuracy(model, dataloader)
    correct = 0
    total = 0
    for (x_batch, y_batch) in dataloader
        preds = onecold(model(x_batch), 0:9)  # Get predicted labels
        truths = onecold(y_batch, 0:9)       # Get true labels
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end

function build_example_cnn()
    layers = []
    push!(layers, Flux.Conv((5,5), 1 => 5, relu))
    push!(layers, Flux.MaxPool((2,2), pad=2))

    push!(layers, Flux.Conv((5,5), 5 => 10, relu))
    push!(layers, Flux.MaxPool((2,2), pad=2))
    push!(layers, BatchNorm(10)) 

    push!(layers, Flux.Conv((5,5), 10 => 15, relu))
    push!(layers, Flux.MaxPool((2,2), pad=2))
    push!(layers, BatchNorm(15)) 
end





    Random.seed!(1)

    X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
    X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]
    
    X_train, y_train = preprocess_greyscale_imagedata(X_train_raw, y_train_raw)
    X_test, y_test = preprocess_greyscale_imagedata(X_test_raw, y_test_raw)

    train_loader = Flux.DataLoader((X_train, y_train), batchsize=128, shuffle=true)
    test_loader = Flux.DataLoader((X_test, y_test), batchsize=128)

    config_nn = Config_Fnn(28 *28, [32], 10, relu)
    model = build_fnn(config_nn)

    loss(y_hat, y) = Flux.crossentropy(y_hat, y)
    ps = Flux.params(model) 
    learning_rate = Float32(0.01)
    opt = Flux.Adam(learning_rate)
    
    custom_train!(model, train_loader, opt, loss, 500)
    basic_train!(model, opt, loss, 100, X_train, y_train)

    y_hat_raw = model(X_test)

    y_hat = onecold(y_hat_raw) .- 1

    y = y_test_raw
    mean(y_hat .== y)
 
    check = [y_hat[i] == y[i] for i in 1:length(y)]

    index = collect(1:length(y))

    check_display = [index y_hat y check]

    vscodedisplay(check_display)

    size(X_train)

function build_example_cnn()
    layers = []
    push!(layers, Flux.Conv((5,5), 1 => 5, relu))
    push!(layers, Flux.MaxPool((2,2), pad=2))

    push!(layers, Flux.Conv((5,5), 5 => 10, relu))
    push!(layers, Flux.MaxPool((2,2), pad=2))
    push!(layers, BatchNorm(10)) 

    push!(layers, Flux.Conv((5,5), 10 => 15, relu))
    push!(layers, Flux.MaxPool((2,2), pad=2))
    push!(layers, BatchNorm(15)) 

    push!(Flux.flatten)

    return Chain(layers...)
end

function run2()
    xs = rand(Float32, 28, 28, 1, 50)
    xs2 = rand(Float32, 10,10,3,7);
    #model = build_example_cnn()
    println(size(xs2))
    #println(size(model(xs)))
    m = Chain(Conv((3,3), 3=>4, pad=1), Flux.flatten, Dense(400,33));
    m2 = Chain(Flux.Conv((5,5), 1 => 5), relu, Flux.flatten)
    println(size(m(xs2)))
    println(size(m2(xs2)))
end

function run3()
    # set random seed
    Random.seed!(1)
    # load data
    X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
    X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]
    println(size(X_train_raw))
end
println("hello world")


function import_bee_data(file_path::String, bee::Bee)
    bee_data_file = joinpath(file_path, "bee_$(bee.id)_data.bson")
    println("bis hierhin") 
    if isfile(bee_data_file)
        BSON.@load bee_data_file begin
            bee.params_history = get(params_history, :params_history, [])
            bee.loss_history = get(loss_history, :loss_history, [])
            bee.accuracy_history = get(accuracy_history, :accuracy_history, [])
        end
    else
        println("Data file for bee $(bee.id) not found.")
    end
end

function import_hive_data(file_path::String, hive::Hive)
    for bee in hive.bee_list
        import_bee_data(file_path, bee)
    end
end


function interact_old(interaction_probability_fn::Function = calculate_interaction_probability, h::Hive, epoch::UInt16)
    accuracy_list = []
    for bee in h.bee_list
        push!(accuracy_list, bee.accuracy_history[epoch])
        bee.interaction_partner_list[epoch] = [0]
    end
    for i in 1:h.n_bees
        accuracy_ratio = accuracy_list[i]/((sum(accuracy_list) - accuracy_list[i])/(h.n_bees - 1))
        interaction_probability = interaction_probability_fn(accuracy_ratio)
        if rand() < interaction_probability #bee with bee_id i is interacting
            interaction_partner = rand(1:h.n_bees)
            if h.bee_list[i].bee_id != i
                println("wrong bee")
            end
            if h.bee_list[i].interaction_partner_list[epoch] == [0]
                h.bee_list[i].interaction_partner_list[epoch] = [interaction_partner]
            end
            if !(interaction_partner in h.bee_list[i].interaction_partner_list[epoch])
                push!(h.bee_list[i].interaction_partner_list[epoch]

"""
the function @choose_partner_extrinsicly calculates the probability of interacting in given @epoch for every @bee in the Hive @h in accordance of the accuracy of the @bee during given epoch.
The accuracy thus has to be already calculated and stored in the accuracy_history
The interaction probability is being calculated by looking at the ratio between the accuracy and the sum of all accuracies of the bees
"""
function choose_partner_extrinsicly!(h::Hive, epoch::UInt16)
    epoch_accuracy_sum = sum(h.accuracy_history[:, epoch])
    for i in 1:h.n_bees
        if h.bee_list[i].bee_id != i
            println("for loop index does not equal bee id")
        end
        interaction_probability = h.accuracy_history[i, epoch]/epoch_accuracy_sum
        if rand() < interaction_probability #bee of bee_id i is interacting
            interaction_partner = rand(1:h.n_bees) #interaction partner is chosen
            h.interaction_partner_history[i, epoch] = interaction_partner
        else 
            h.interaction_partner_history[i, epoch] = 0
        end
    end
    return 0
end

function export_bee_data(file_path::String, bee::Bee)
    mkpath(file_path)
    bee_data_file = joinpath(file_path, "bee_$(bee.id)_data.bson")
    BSON.@save bee_data_file params_history=bee.params_history loss_history=bee.loss_history accuracy_history=bee.accuracy_history
end

function export_hive_data(file_path::String, hive::Hive)
    for bee in hive.bee_list
        export_bee_data(file_path, bee)
    end
end

function import_bson_test(file_path::String, id::Int)
    bee_data_file = joinpath(file_path, "bee_$(id)_data.bson")
    if isfile(bee_data_file)
        BSON.@load bee_data_file params_history
        println("loss_history: ")
        println(loss_history)
    else
        println("does not exits")
    end
end