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


"""
version of calc_accuracy in case the output layer dimension is equal to the sum of the labels of the various tasks
"""
function calc_task_accuracy(model, dataloader, output_range, num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = Flux.onecold(model(x_batch)[output_range[1]:output_range[2]], 0:(output_range[2] - output_range[1]))  # Get predicted labels
        truths = Flux.onecold(y_batch, 0:n_labels - 1)       # Get true labels
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end



"""
This will be the new version of @train_task! that will take into account that the different tasks will have different dimensions
"""
 
function learn_task!(h::Hive, data, output_range, n_epochs::UInt16 = DEFAULTS[:N_EPOCHS])
    learning_rate = DEFAULTS[:LEARNING_RATE]
    optimizer = Flux.Adam(learning_rate) 
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
    trainloader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((data[3], data[4]), batchsize=128)
    for bee in h.bee_list
        #initial_accuracy = calc_accuracy(bee.brain, testloader, n_labels)
        #h.accuracy_history[bee.id, 1] = initial_accuracy
    end
    for epoch = 1:n_epochs
        h.epoch_index += UInt(1)
        for bee in h.bee_list
            epoch_loss = 0.0
            for (x_batch, y_batch) in trainloader
                model = bee.brain
                grads = gradient(()->loss_fn(model(x_batch)[output_range[1]:output_range[2]], y_batch), Flux.params(model))
                Flux.Optimise.update!(optimizer, Flux.params(model), grads)
                epoch_loss += loss_fn(model(x_batch), y_batch)
            end
            h.loss_history[bee.id, epoch] = epoch_loss
            #accuracy = calc_accuracy(bee.brain, testloader, n_labels)
            #h.accuracy_history[bee.id, (epoch + 1)] = accuracy
            bee.params_history[epoch] = deepcopy(Flux.params(bee.brain))
            println("Epoch = $epoch : Bee ID = $(bee.id) : Loss = $epoch_loss : Accuracy = $accuracy")
        end
    end
    #save_data(RAW_PATH, h, n_epochs)
    return 0
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

"""
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



function prepare_mnist_dataset(split::Symbol; subset=nothing)
    dataset = MLDatasets.MNIST(split)
    images = dataset.features
    labels = dataset.targets .+ 1 # Ensure 1-based indexing for Flux.onehotbatch

    num_samples = subset !== nothing ? subset : size(images, 3)

    processed_images = Array{Float32, 4}(undef, 32, 32, 3, num_samples)
    processed_images = map(preprocess_greyscale_28x28, eachslice(images; dims=3))
    processed_images = permutedims(cat(processed_images...; dims=4), (1,2,3,4))

    onehot_labels = Flux.onehotbatch(labels, 1:10)

    
    return processed_images, onehot_labels
end

function preprocess_greyscale_28x28(image)
    img = Float32.(image) / 255.0
    img_padded = padarray(img, Fill(0, (2,2)))
    img_rgb = cat(img_padded, img_padded, img_padded; dims=3)
    return img_rgb
end


"""
The function @train_bee! trains one given neural network (Bee) by iterating once over the data given in the @dataloader. 
"""
function train_model_old!(model, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y)
    optimizer = Flux.Adam(learning_rate)

    total_loss = 0.0
    n_batches = 0

    for (x_batch, y_batch) in dataloader
        loss = loss_fn(model(x_batch), y_batch)
        grads = gradient(()->loss_fn(model(x_batch), y_batch), Flux.params(model))
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)
        total_loss += loss
    end
    return total_loss
end

function build_cifar10_model_old()

    return Chain(
        # Convolutional Block 1
        Conv((3, 3), 3 => 32, relu, pad=1), BatchNorm(32), Dropout(0.2),
        Conv((3, 3), 32 => 32, relu, pad=1), BatchNorm(32), MaxPool((2, 2)),

        # Convolutional Block 2
        Conv((3, 3), 32 => 64, relu, pad=1), BatchNorm(64), Dropout(0.3),
        Conv((3, 3), 64 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),

        # Convolutional Block 3
        Conv((3, 3), 64 => 128, relu, pad=1), BatchNorm(128), Dropout(0.4),
        Conv((3, 3), 128 => 128, relu, pad=1), BatchNorm(128), MaxPool((2, 2)),

        # Flatten and Fully Connected Layers
        Flux.flatten,
        Dense(128 * 4 * 4, 512, relu), Dropout(0.5),
        Dense(512, 256, relu), Dropout(0.5),
        Dense(256, 10)  # CIFAR-10 has 10 classes
    )
end


function build_brain(; input_size::AbstractVector{<:Integer}=DEFAULTS[:INPUT_SIZE], output_size::UInt16=DEFAULTS[:OUTPUT_SIZE])
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
    #the softmax seems to be unnecessary since Flux.crossentropy applies it internally
    #push!(layers, Flux.softmax)
    return Chain(layers...)
end


function build_brain_2(; input_size::AbstractVector{<:Integer}=DEFAULTS[:INPUT_SIZE], output_size::UInt16=DEFAULTS[:OUTPUT_SIZE])
    layers = []
    push!(layers, Flux.Conv((3,3), input_size[3] => 10, pad=1, relu))
    push!(layers, Flux.Conv((3,3), 10 => 10, pad=1, relu))
    push!(layers, Flux.MaxPool((2,2), stride=2))

    push!(layers, Flux.Conv((3,3), 10 => 10, pad=1, relu))
    push!(layers, Flux.Conv((3,3), 10 => 10, pad=1, relu))
    push!(layers, Flux.MaxPool((2,2), stride=2))
    push!(layers, BatchNorm(10)) 

    push!(layers, Flux.Conv((3,3), 10 => 5, pad=1, relu))
    push!(layers, Flux.Conv((3,3), 5 => 5, pad=1, relu))
    push!(layers, Flux.MaxPool((2,2), stride=2))

    @show Flux.outputsize(Chain(layers[1:end-3]...), (input_size[1], input_size[2], input_size[3], 1))

    push!(layers, Flux.flatten)
    input_size_dense = Int64((input_size[1]/8) * (input_size[2]/8) * 5)
    push!(layers, Flux.Dense(input_size_dense, 40, relu))
    push!(layers, Flux.Dense(40, 20, relu))
    push!(layers, Flux.Dense(20, output_size))
    #the softmax seems to be unnecessary since Flux.crossentropy applies it internally
    #push!(layers, Flux.softmax)
    return Chain(layers...)
end

function build_cifar10_model(; input_size::AbstractVector{<:Integer}=DEFAULTS[:INPUT_SIZE], output_size::UInt16=DEFAULTS[:OUTPUT_SIZE])

    return Chain(
        # Convolutional Block 1
        Conv((3, 3), input_size[3] => 32, relu, pad=1), BatchNorm(32), Dropout(0.2),
        Conv((3, 3), 32 => 32, relu, pad=1), BatchNorm(32), MaxPool((2, 2)),

        # Convolutional Block 2
        Conv((3, 3), 32 => 64, relu, pad=1), BatchNorm(64), Dropout(0.3),
        Conv((3, 3), 64 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),

        # Convolutional Block 3
        Conv((3, 3), 64 => 128, relu, pad=1), BatchNorm(128), Dropout(0.4),
        Conv((3, 3), 128 => 128, relu, pad=1), BatchNorm(128), MaxPool((2, 2)),

        # Flatten and Fully Connected Layers
        Flux.flatten,
        Dense(128 * (input_size[1]/8) * (input_size[1]/8), 512, relu), Dropout(0.5),
        Dense(512, 256, relu), Dropout(0.5),
        Dense(256, output_size)  # CIFAR-10 has 10 classes
    )
end


function build_cifar10_model_small(; input_size::AbstractVector{<:Integer}=DEFAULTS[:INPUT_SIZE], output_size::UInt16=DEFAULTS[:OUTPUT_SIZE])

    return Chain(
        # Convolutional Block 1
        Conv((3, 3), input_size[3] => 16, relu, pad=1), BatchNorm(16), Dropout(0.2),
        Conv((3, 3), 16 => 16, relu, pad=1), BatchNorm(16), MaxPool((2, 2)),

        # Convolutional Block 2
        Conv((3, 3), 16 => 32, relu, pad=1), BatchNorm(32), Dropout(0.3),
        Conv((3, 3), 32 => 32, relu, pad=1), BatchNorm(32), MaxPool((2, 2)),

        # Convolutional Block 3
        Conv((3, 3), 32 => 64, relu, pad=1), BatchNorm(64), Dropout(0.4),
        Conv((3, 3), 64 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),

        # Flatten and Fully Connected Layers
        Flux.flatten,
        Dense(64 * Int64(input_size[1]/8) * Int64(input_size[2]/8), 512, relu), Dropout(0.5),
        Dense(512, 256, relu), Dropout(0.5),
        Dense(256, output_size)  # CIFAR-10 has 10 classes
    )
end

function build_model_3(; input_size::AbstractVector{<:Integer}=DEFAULTS[:INPUT_SIZE], output_size::UInt16=DEFAULTS[:OUTPUT_SIZE])

    return Chain(
        # Convolutional Block 1
        Conv((3, 3), input_size[3] => 64, relu, pad=1), BatchNorm(64), Dropout(0.2),
        Conv((3, 3), 64 => 64, relu, pad=1), BatchNorm(64), MaxPool((2, 2)),

        # Convolutional Block 2
        Conv((3, 3), 64 => 128, relu, pad=1), BatchNorm(128), Dropout(0.3),
        Conv((3, 3), 128 => 128, relu, pad=1), BatchNorm(128), MaxPool((2, 2)),

        # Convolutional Block 3
        Conv((3, 3), 128 => 128, relu, pad=1), BatchNorm(128), Dropout(0.4),
        Conv((3, 3), 128 => 128, relu, pad=1), BatchNorm(128), MaxPool((2, 2)),

        # Flatten and Fully Connected Layers
        Flux.flatten,
        Dense(128 * Int64(input_size[1]/8) * Int64(input_size[1]/8), 512, relu), Dropout(0.5),
        Dense(512, 256, relu), Dropout(0.5),
        Dense(256, output_size)  # CIFAR-10 has 10 classes
    )
end


"""
the function @prepare_MNIST loads the MNIST dataset from the @MLDatasets package and returns it in an array 
    of the form [x_test, y_test, x_train, y_train].
    Via @use_subset one can also choose to only use a subset of the MNIST dataset for quick testruns.
"""
function prepare_MNIST(normalize::Bool=false, use_subset::Bool=true, subset_train::UInt16=UInt16(6000), subset_test::UInt16=UInt16(1000))
    x_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
    x_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]
    if use_subset == true
        x_train_raw = x_train_raw[:, :, 1:subset_train]
        y_train_raw = y_train_raw[1:subset_train]
        x_test_raw = x_test_raw[:, :, 1:subset_test]
        y_test_raw = y_test_raw[1:subset_test]
    end
    x_train_raw_size = size(x_train_raw)
    x_test_raw_size = size(x_test_raw)
    x_train = reshape(x_train_raw, x_train_raw_size[1], size(x_train_raw)[2], 1, size(x_train_raw)[3])
    x_test = reshape(x_test_raw, x_test_raw_size[1], size(x_test_raw)[2], 1, size(x_test_raw)[3])
    y_train = Flux.onehotbatch(y_train_raw, 0:9)
    y_test = Flux.onehotbatch(y_test_raw, 0:9)
    if normalize == true
        x_train_raw = x_test_raw ./ 255.0
        x_test_raw = x_test_raw ./ 255.0
    end
    data = [x_train, y_train, x_test, y_test]
    return data
end

function prepare_CIFAR10(normalize::Bool=false, use_subset::Bool=true, subset_train::UInt16=UInt16(6000), subset_test::UInt16=UInt16(1000))
    x_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
    x_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]
    data = prepare_data(x_train_raw, y_train_raw, x_test_raw, y_test_raw, 10, normalize, use_subset, subset_train, subset_test)
    return data
end

function prepare_data(x_train_raw, y_train_raw, x_test_raw, y_test_raw, n_labels, normalize::Bool=false, use_subset::Bool=true, subset_train::UInt16=UInt16(6000), subset_test::UInt16=UInt16(1000))
    if use_subset == true
        x_train_raw = x_train_raw[:, :, 1:subset_train]
        y_train_raw = y_train_raw[1:subset_train]
        x_test_raw = x_test_raw[:, :, 1:subset_test]
        y_test_raw = y_test_raw[1:subset_test]
    end
    x_train_raw_size = size(x_train_raw)
    x_test_raw_size = size(x_test_raw)
    x_train = reshape(x_train_raw, x_train_raw_size[1], size(x_train_raw)[2], 1, size(x_train_raw)[3])
    x_test = reshape(x_test_raw, x_test_raw_size[1], size(x_test_raw)[2], 1, size(x_test_raw)[3])
    y_train = Flux.onehotbatch(y_train_raw, 0:(n_labels - 1))
    y_test = Flux.onehotbatch(y_test_raw, 0:(n_labels - 1))
    if normalize == true
        x_train_raw = x_test_raw ./ 255.0
        x_test_raw = x_test_raw ./ 255.0
    end
    data = [x_train, y_train, x_test, y_test]
    return data
end

"""
the function @calc_accuracy calculates the accuracy of a neural network given by @model by averaging the results of the model on the dataset given in @dataloader
"""
function calc_accuracy_labels(model, dataloader; n_labels = 10, num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = Flux.onecold(model(x_batch), 0:(n_labels - 1))  # Get predicted labels
        truths = Flux.onecold(y_batch, 0:n_labels - 1)       # Get true labels
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end


"""
takes Hive, max_time, lambda_Train and lambda_Interact
"""
function gillespie_train_task!(h::Hive, max_time, lambda_Train, lambda_Interact)
    time = 0.0

    while time < max_time
        a_train = lambda_Train * h.n_bees #propensity for training the neural networks. All networks (Bees) train with the same rate @lambda_Train
        a_interact = lambda_Interact * h.n_bees #this is a dummy propensity. normally the rate at which an interaction takes place is dependent on the current accuracies of the networks
        total_propensity = a_train + a_interact
        println(total_propensity)

        d_t = -log(rand()) / total_propensity
        println(d_t)
        time = time + d_t

        choose_action = rand() * total_propensity
        if choose_action < a_train
            selected_bee = h.bee_list[rand(1:h.n_bees)]
            println("hier wird dann die eine gewählte bee mit id $(selected_bee.id) trainiert")
        else
            println("hier würde man durch ene liste an interactions durchgehen, bis die summe dieser raten unsere random chosen variable übersteigt")
        end
    end
end

function train_task!(h::Hive, data, n_epochs::UInt16 = DEFAULTS[:N_EPOCHS])
    learning_rate = DEFAULTS[:LEARNING_RATE]
    optimizer = Flux.Adam(learning_rate) 
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
    trainloader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((data[3], data[4]), batchsize=128)
    for bee in h.bee_list
        initial_accuracy = calc_accuracy(bee.brain, testloader)
        h.accuracy_history[bee.id, 1] = initial_accuracy
    end
    for epoch = 1:n_epochs
        h.epoch_index += UInt(1)
        for bee in h.bee_list
            epoch_loss = 0.0
            for (x_batch, y_batch) in trainloader
                model = bee.brain
                grads = gradient(()->loss_fn(model(x_batch), y_batch), Flux.params(model))
                Flux.Optimise.update!(optimizer, Flux.params(model), grads)
                epoch_loss += loss_fn(model(x_batch), y_batch)
            end
            h.loss_history[bee.id, epoch] = epoch_loss
            accuracy = calc_accuracy(bee.brain, testloader)
            h.accuracy_history[bee.id, (epoch + 1)] = accuracy
            bee.params_history[epoch] = deepcopy(Flux.params(bee.brain))
            println("Epoch = $epoch : Bee ID = $(bee.id) : Loss = $epoch_loss : Accuracy = $accuracy")
        end
    end
    save_data(RAW_PATH, h, n_epochs)
    return 0
end

function run_gillespie(; n_epochs=N_EPOCHS, n_steps_per_epoch=DEFAULTS[:N_STEPS_PER_EPOCH], dataset_function=prepare_mnist_dataset_1_channel)
    h = Hive(:Val{classification}, N_BEES, N_EPOCHS, brain_constructor = build_model_4)
    train_features_mnist_subset, train_labels_mnist_subset = dataset_function(:train)
    test_features_mnist_subset, test_labels_mnist_subset = dataset_function(:test)
    trainloader = Flux.DataLoader((train_features_mnist_subset, train_labels_mnist_subset), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((test_features_mnist_subset, test_labels_mnist_subset), batchsize=128, shuffle=true)
    gillespie_train_task_with_epochs!(h, n_epochs=n_epochs, trainloader=trainloader, testloader=testloader, n_steps_per_epoch=n_steps_per_epoch, lambda_Interact=0.0)
    println(h.propensity_ratio_history)
end
