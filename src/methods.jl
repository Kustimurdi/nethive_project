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


"""
the function @calc_accuracy calculates the accuracy of a neural network given by @model by averaging the results of the model on the dataset given in @dataloader
"""
function calc_accuracy(model, dataloader; num_batches::Int=typemax(Int))
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

function save_data(raw_path::String, h::Hive, n_epochs=DEFAULTS[:N_EPOCHS])
    mkpath(raw_path)
    #epoch_ids = collect(1:h.epoch_index)
    epoch_ids = collect(1:n_epochs)
    #accuracies_epoch_ids = collect(0:n_epochs)
    export_data(string(raw_path, "/loss_history", ".csv"), h.loss_history, h.n_bees, epoch_ids, "loss")
    export_data(string(raw_path, "/n_subdom_interactions_history", ".csv"), h.n_subdom_interactions_history, h.n_bees, epoch_ids, "n_subdominant_interactions")
    export_data(string(raw_path, "/n_dom_interactions_history", ".csv"), h.n_subdom_interactions_history, h.n_bees, epoch_ids, "n_dominant_interactions")
    export_data(string(raw_path, "/accuracy_history", ".csv"), h.accuracy_history, h.n_bees, epoch_ids, "accuracy")
    return 0
end

function save_nn_state(raw_net_path::String, h::Hive)
    mkpath(raw_net_path)
    brains = [h.bee_list[i].brain for i in 1:h.n_bees]

    serialize(string(raw_net_path, "epoch_", h.epoch_index, ".brains"), brains)
    return nothing
end

"""
the function @compute_K_matrix calculates the interaction propensities for all pairs of @Bee objects in accordance to the interaction rate stated in the wasp paper 
paramteres: 
- current_accuracies_list: a list that holds the current accuracies of the neural networks of the @Bee objects (the accuracy is a proxy for the queen gene expression)
- lambda_Interact: a parameter for the sigmoid function that gives the probability of bee_i being in a subdominant interaction with bee_j
"""
function compute_K_matrix(current_accuracies_list; lambda_Interact=DEFAULTS[LAMBDA_INTERACT])
    K_matrix = K_func.(current_accuracies_list, current_accuracies_list', lambda_Interact)
    K_matrix[diagind(K_matrix)] .= 0
    #K_matrix -= Diagonal(K_matrix)
    return K_matrix 
end

function K_func(r_i, r_j, lambda) 
    return r_i*r_j*(1-Flux.sigmoid(-lambda*(r_i - r_j)))
end

function gillespie_train_task_with_epochs!(h::Hive; trainloader, testloader, n_epochs=DEFAULTS[:N_EPOCHS], lambda_Train=DEFAULTS[:LAMBDA_TRAIN], lambda_Interact=DEFAULTS[:LAMBDA_INTERACT], n_steps_per_epoch=DEFAULTS[:N_STEPS_PER_EPOCH])
    total_elapsed_time = 0.0
    gillespie_time = Float64(0.0)
    for bee in h.bee_list
        initial_accuracy = calc_accuracy(bee.brain, testloader)
        h.initial_accuracies_list[bee.id] = initial_accuracy
        h.current_accuracies_list[bee.id] = initial_accuracy
        bee.current_accuracy = initial_accuracy
    end
    for epoch in 1:n_epochs
        epoch_start_time = time()
        @info "Starting epoch $(epoch)"
        n_actions = 0
        n_train =0
        n_interact = 0
        n_dom_higher_acc = 0
        n_sub_higher_acc = 0
        while gillespie_time < epoch

            epoch_loop_time = time()
            a_train = lambda_Train * h.n_bees #propensity for training the neural networks. All networks (Bees) train with the same rate @lambda_Train
            K_matrix = compute_K_matrix(h.current_accuracies_list, lambda_Interact=lambda_Interact)
            a_interact = sum(K_matrix)

            total_propensity = a_train + a_interact

            d_t = rand(Exponential(1 / (n_steps_per_epoch * h.n_bees)))
            gillespie_time += d_t
            #println("gillespie time: $(gillespie_time)")

            choose_action = rand() * total_propensity
            if choose_action < a_train

                selected_bee = h.bee_list[rand(1:h.n_bees)]
                loss = train_model!(selected_bee.brain, trainloader, learning_rate=DEFAULTS[:LEARNING_RATE])

                h.loss_history[selected_bee.id, epoch] += loss
                current_accuracy = calc_accuracy(selected_bee.brain, testloader)
                h.current_accuracies_list[selected_bee.id] = current_accuracy
                selected_bee.current_accuracy = current_accuracy

                n_train +=1
                println("n train: $(n_train)")
                println("trained bee = $(selected_bee.id) : current accuracy = $(selected_bee.current_accuracy)")
            else
                #!!!!!!!!!!!!hier weiter implimentieren
                sub_bee, dom_bee = choose_interaction(h, a_interact, K_matrix)
                h.n_subdom_interactions_history[sub_bee.id, epoch] += 1
                h.n_dom_interactions_history[dom_bee.id, epoch] +=1

                if dom_bee.current_accuracy >= sub_bee.current_accuracy
                    n_dom_higher_acc+=1
                else
                    n_sub_higher_acc+=1
                end

                println("sub bee is $(sub_bee.id) : current accuracy=$(sub_bee.current_accuracy)")
                println("dom bee is $(dom_bee.id) : current accuracy=$(dom_bee.current_accuracy)")

                n_interact+=1
                println("n interact: $(n_interact)")
            end
            n_actions +=1
            epoch_loop_elapsed_time = time() - epoch_loop_time
            @info "Epoch $(epoch) loop $(n_actions) completed" propensity_ratio=(a_train/a_interact) loop_time=epoch_loop_elapsed_time
        end
        h.accuracy_history[:, epoch] = h.current_accuracies_list[:, 1]
        elapsed_time = time() - epoch_start_time
        total_elapsed_time += elapsed_time
        @info "Epoch $(epoch) completed" epoch=epoch elapsed_time=elapsed_time gillespie_time=gillespie_time average_accuracy=mean(h.current_accuracies_list[:,1]) n_actions=n_actions n_train=n_train n_interact=n_interact
        @info "Memory usage $(Sys.total_memory()) bytes"
        save_nn_state(RAW_NET_PATH, h)
        h.epoch_index+=1
    end
    save_data(RAW_PATH, h, n_epochs)
    @info "Gillespie simulation is over. Data path: $(RAW_PATH)" total_elapsed_time=total_elapsed_time
end

"""
muss fertig implimentiert werden
"""
function choose_interaction(h::Hive, a_interact, K_matrix)
    choose_interaction = rand() * a_interact
    cumulative_sum = 0
    for i in 1:h.n_bees
        for j in 1:h.n_bees
            cumulative_sum += K_matrix[i, j]
            if cumulative_sum >= choose_interaction
                return (h.bee_list[i],h.bee_list[j])
            end
        end 
    end 
end


function train_model!(model, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
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

function run_gillespie(; n_epochs=N_EPOCHS, n_steps_per_epoch=DEFAULTS[:N_STEPS_PER_EPOCH], dataset_function=prepare_mnist_dataset_1_channel)
    h = Hive(N_BEES, N_EPOCHS, brain_constructor = build_model_4)
    train_features_mnist_subset, train_labels_mnist_subset = dataset_function(:train)
    test_features_mnist_subset, test_labels_mnist_subset = dataset_function(:test)
    trainloader = Flux.DataLoader((train_features_mnist_subset, train_labels_mnist_subset), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((test_features_mnist_subset, test_labels_mnist_subset), batchsize=128, shuffle=true)
    gillespie_train_task_with_epochs!(h, n_epochs=n_epochs, trainloader=trainloader, testloader=testloader, n_steps_per_epoch=n_steps_per_epoch, lambda_Interact=0.0)
end

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
