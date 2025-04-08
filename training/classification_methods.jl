function prepare_mnist_dataset_1_channel(split::Symbol; subset=nothing)
    dataset = MLDatasets.MNIST(split)
    processed_images, onehot_labels = prepare_28x28_dataset(dataset; subset=subset)
    return processed_images, onehot_labels
end

function prepare_fashionmnist_dataset_1_channel(split::Symbol; subset=nothing)
    dataset = MLDatasets.FashionMNIST(split)
    processed_images, onehot_labels = prepare_28x28_dataset(dataset; subset=subset)
    return processed_images, onehot_labels
end


function prepare_28x28_dataset(dataset; subset=nothing)
    images = dataset.features
    images = Float32.(images) / 255.0
    labels = dataset.targets .+ 1  # Ensure 1-based indexing

    dataset_wide_mean_img = mean(images)
    dataset_wide_std_img = std(images)
    images = (images .- dataset_wide_mean_img) ./ dataset_wide_std_img

    num_samples = subset !== nothing ? subset : size(images, 3)
    
    # Preallocate an array instead of concatenating
    processed_images = Array{Float32, 4}(undef, 32, 32, 1, num_samples)

    # Process each image and store it in the preallocated array
    for i in 1:num_samples
        img_padded = parent(padarray(images[:, :, i], Fill(0, (2, 2))))  # Zero-pad to 32x32
        processed_images[:, :, 1, i] .= img_padded  
    end

    # Convert labels to one-hot encoding
    onehot_labels = Flux.onehotbatch(labels[1:num_samples], 1:10)

    return processed_images, onehot_labels
end

function rgb_to_gray(imgs)
    greyscale = 0.299 .* imgs[:, :, 1, :] .+ 
                0.587 .* imgs[:, :, 2, :] .+ 
                0.114 .* imgs[:, :, 3, :]
    return reshape(greyscale, 32, 32, 1, size(imgs, 4))
end

function prepare_cifar10_dataset_greyscale(split::Symbol; subset=nothing)
    dataset = MLDatasets.CIFAR10(split)
    images = rgb_to_gray(dataset.features)
    images = Float32.(images) / 255.0
    labels = dataset.targets .+ 1

    dataset_wide_mean_img = mean(images)
    dataset_wide_std_img = std(images)
    images = (images .- dataset_wide_mean_img) ./ dataset_wide_std_img

    onehot_labels = Flux.onehotbatch(labels, 1:10)

    if subset !== nothing
        images = images[:, :, :, 1:subset]
        onehot_labels = onehot_labels[:, 1:subset]
    end
    
    return images, onehot_labels
end

function prepare_svhn2_dataset_greyscale(split::Symbol; subset=nothing)
    dataset = MLDatasets.SVHN2(split)
    images = rgb_to_gray(dataset.features)
    images = Float32.(images) / 255.0
    labels = dataset.targets

    dataset_wide_mean_img = mean(images)
    dataset_wide_std_img = std(images)
    images = (images .- dataset_wide_mean_img) ./ dataset_wide_std_img

    onehot_labels = Flux.onehotbatch(labels, 1:10)

    if subset !== nothing
        images = images[:, :, :, 1:subset]
        onehot_labels = onehot_labels[:, 1:subset]
    end
    
    return images, onehot_labels
end

function calc_classification_accuracy(model, dataloader; num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = Flux.onecold(model(x_batch), 0:9)  # Get predicted labels
        truths = Flux.onecold(y_batch, 0:9)       # Get true labels
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end


function train_classification_model!(model, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
    loss_fn(x, y) = Flux.logitcrossentropy(model(x), y)
    optimizer = Flux.Adam(learning_rate)

    total_batch_loss = 0.0
    n_batches = 0

    for (x_batch, y_batch) in dataloader
        Flux.train!(loss_fn, Flux.params(model), [(x_batch, y_batch)], optimizer)

        total_batch_loss += loss_fn(x_batch, y_batch)
        n_batches += 1
    end

    return total_batch_loss / n_batches  # Return average loss
end

"""
------------------------------------------------------------
for training and testing neural network architectures
------------------------------------------------------------
"""

function train_neural_network(raw_path, model, n_epochs; trainloader, testloader, learning_rate=DEFAULTS[:LEARNING_RATE])
    #@info("Path for saving data: $(data_path)")
    mkpath(raw_path)
    loss_history = zeros(Float32, 1, n_epochs)
    accuracy_history = zeros(Float32, 1, n_epochs)
    total_elapsed_time = 0.0
    for epoch in 1:n_epochs
        @info "Starting epoch $(epoch)"
        start_time = time()
        loss = train_model!(model, trainloader, learning_rate=learning_rate)
        accuracy = calc_accuracy(model, testloader)
        loss_history[epoch] = loss
        accuracy_history[epoch] = accuracy
        elapsed_time = time() - start_time
        total_elapsed_time += elapsed_time
        @info "Epoch $(epoch) completed" epoch=epoch time=elapsed_time loss=loss accuracy=accuracy
        @info "Memory usage $(Sys.total_memory()) bytes"
    end
    @info "Done. Total elapsed time is $(total_elapsed_time)"
    @info("Path for saving data: $(raw_path)")
    epoch_ids = collect(1:n_epochs)
    export_data(string(raw_path, "/loss_history.csv"), loss_history, 1, epoch_ids, "loss")
    export_data(string(raw_path, "/accuracy_history.csv"), accuracy_history, 1, epoch_ids, "accuracy")
end

function run_training(model; dataset_function)
    if dataset_function==prepare_mnist_dataset_1_channel
        dataset_name = "test_mnist"
    elseif dataset_function==prepare_cifar10_dataset_greyscale
        dataset_name = "test_cifar10"
    elseif dataset_function==prepare_fashionmnist_dataset_1_channel
        dataset_name = "test_fashion"
    elseif dataset_function==prepare_svhn2_dataset_greyscale
        dataset_name = "test_svhn2"
    else 
        throw(ErrorException("Incorrect dataset_function was provided"))
    end
    dataset_name = string(dataset_name, "_learning_rate_", LEARNING_RATE, Dates.format(now(), "DyymmddTHHMMSSss"), "I", rand(1:9, 1)[1])
    data_path::String = string("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/", PARENT_DATASET_NAME, "/", dataset_name)
    x_train, y_train = dataset_function(:train)
    x_test, y_test = dataset_function(:test)
    trainloader = Flux.DataLoader((x_train, y_train), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((x_test, y_test), batchsize=128, shuffle=true)
    train_neural_network(data_path, model, N_EPOCHS, trainloader=trainloader, testloader=testloader, learning_rate=LEARNING_RATE)
end

