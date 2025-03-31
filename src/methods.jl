"""
the function @calc_accuracy calculates the accuracy of a neural network given by @model by averaging the results of the model on the dataset given in @dataloader
"""
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

function calc_regression_accuracy(model, dataloader; atol=0.005, num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = model(x_batch)
        truths = y_batch       # Get true labels
        result = isapprox.(vec(preds), vec(truths), atol=atol)

        num_true = count(result)
        num_false = count(!, result)

        correct += num_true
        total += length(truths)
    end
    return correct / total
end



"""
the function @compute_K_matrix calculates the interaction propensities for all pairs of @Bee objects in accordance to the interaction rate stated in the wasp paper 
paramteres: 
- current_accuracies_list: a list that holds the current accuracies of the neural networks of the @Bee objects (the accuracy is a proxy for the queen gene expression)
- lambda_Interact: a parameter for the sigmoid function that gives the probability of bee_i being in a subdominant interaction with bee_j
"""
function compute_K_matrix(current_accuracies_list; lambda_interact=DEFAULTS[LAMBDA_INTERACT])
    K_matrix = K_func.(current_accuracies_list, current_accuracies_list', lambda_interact)
    K_matrix[diagind(K_matrix)] .= 0
    return K_matrix 
end

function K_func(r_i, r_j, lambda) 
    return r_i*r_j*Flux.sigmoid(-lambda*(r_i - r_j))
end

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

function punish_regression_model!(model, dataloader, punish_rate)
    loss_fn(x, y) = Flux.Losses.mse(model(x), y)
    total_batch_loss = 0.0
    n_batches = 0
    for (x_batch, y_batch) in dataloader
        grads = Flux.gradient(() -> loss_fn(x_batch, y_batch), Flux.params(model))
        for p in Flux.params(model)
            p .= p .+ punish_rate .* grads[p]
        end

        total_batch_loss += loss_fn(x_batch, y_batch)
        n_batches +=1
    end
    return total_batch_loss/n_batches
end

function train_regression_model!(model, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
    loss_fn(x, y) = Flux.Losses.mse(model(x), y)
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


function gillespie_regression!(h::Hive; trainloader, testloader, n_epochs=DEFAULTS[:N_EPOCHS], learning_rate=DEFAULTS[:LEARNING_RATE], punish_rate=DEFAULTS[:PUNISH_RATE], acc_atol=DEFAULTS[:ACCURACY_ATOL], lambda_train=DEFAULTS[:LAMBDA_TRAIN], lambda_interact=DEFAULTS[:LAMBDA_INTERACT], n_steps_per_epoch=DEFAULTS[:N_STEPS_PER_EPOCH])
    total_elapsed_time = 0.0
    gillespie_time = Float64(0.0)
    for bee in h.bee_list
        initial_accuracy = calc_regression_accuracy(bee.brain, testloader, atol=acc_atol)
        h.initial_accuracies_list[bee.id] = initial_accuracy
        h.current_accuracies_list[bee.id] = initial_accuracy
        bee.current_accuracy = initial_accuracy
    end
    save_nn_state(RAW_NET_PATH, h)
    for epoch in 1:n_epochs
        epoch_start_time = time()
        @info "Starting epoch $(epoch)"
        n_actions = 0
        n_train =0
        n_interact = 0
        while gillespie_time < epoch

            loop_start_time = time()
            a_train = lambda_train * h.n_bees #propensity for training the neural networks. All networks (Bees) train with the same rate @lambda_Train
            K_matrix = compute_K_matrix(h.current_accuracies_list, lambda_interact=lambda_interact)
            a_interact = sum(K_matrix)

            total_propensity = a_train + a_interact

            d_t = rand(Exponential(1 / (n_steps_per_epoch * h.n_bees)))
            gillespie_time += d_t

            choose_action = rand() * total_propensity
            if choose_action < a_train

                selected_bee = h.bee_list[rand(1:h.n_bees)]
                loss = train_regression_model!(selected_bee.brain, trainloader, learning_rate=learning_rate)

                h.loss_history[selected_bee.id, epoch] += loss
                current_accuracy = calc_regression_accuracy(selected_bee.brain, testloader, atol=acc_atol)
                h.current_accuracies_list[selected_bee.id] = current_accuracy
                selected_bee.current_accuracy = current_accuracy

                h.n_train_history[selected_bee.id, epoch] += 1
                n_train +=1
                println("n train: $(n_train)")
                println("trained bee = $(selected_bee.id) : current accuracy = $(selected_bee.current_accuracy)")
            else
                sub_bee, dom_bee = choose_interaction(h, a_interact, K_matrix)

                punish_regression_model!(sub_bee.brain, trainloader, punish_rate)
                new_accuracy = calc_regression_accuracy(sub_bee.brain, testloader, atol=acc_atol)
                println("bee id = $(sub_bee.id) : old acc = $(sub_bee.current_accuracy) : new acc = $(new_accuracy)")

                h. current_accuracies_list[sub_bee.id] = new_accuracy
                sub_bee.current_accuracy = new_accuracy

                h.n_subdom_interactions_history[sub_bee.id, epoch] += 1
                h.n_dom_interactions_history[dom_bee.id, epoch] +=1

                n_interact+=1
                println("n interact: $(n_interact)")
            end
            n_actions +=1
            epoch_loop_elapsed_time = time() - loop_start_time
            @info "Epoch $(epoch) loop $(n_actions) completed" propensity_ratio=(a_train/a_interact) loop_time=epoch_loop_elapsed_time a_train=a_train a_interact=a_interact
        end
        h.epoch_index+=1
        h.accuracy_history[:, epoch] = h.current_accuracies_list[:, 1]
        elapsed_time = time() - epoch_start_time
        total_elapsed_time += elapsed_time
        @info "Epoch $(epoch) completed" epoch=epoch elapsed_time=elapsed_time gillespie_time=gillespie_time average_accuracy=mean(h.current_accuracies_list[:,1]) n_actions=n_actions n_train=n_train n_interact=n_interact 
        @info "Memory usage $(Sys.total_memory()) bytes"
        save_nn_state(RAW_NET_PATH, h)
    end
    save_data(RAW_PATH, h, n_epochs)
    @info "Gillespie simulation is over. Data path: $(RAW_PATH)" total_elapsed_time=total_elapsed_time
end

function save_data(raw_path::String, h::Hive, n_epochs=DEFAULTS[:N_EPOCHS])
    mkpath(raw_path) 
    #epoch_ids = collect(1:h.epoch_index) 
    epoch_ids = collect(1:n_epochs)
    #accuracies_epoch_ids = collect(0:n_epochs)
    export_data(string(raw_path, "/accuracy_history", ".csv"), h.accuracy_history, h.n_bees, epoch_ids, "accuracy")
    export_data(string(raw_path, "/loss_history", ".csv"), h.loss_history, h.n_bees, epoch_ids, "loss")
    export_data(string(raw_path, "/train_history", ".csv"), h.n_train_history, h.n_bees, epoch_ids, "n_train")
    export_data(string(raw_path, "/subdom_interactions_history", ".csv"), h.n_subdom_interactions_history, h.n_bees, epoch_ids, "n_subdominant_interactions")
    export_data(string(raw_path, "/dom_interactions_history", ".csv"), h.n_dom_interactions_history, h.n_bees, epoch_ids, "n_dominant_interactions")
    return 0
end

function save_nn_state(raw_net_path::String, h::Hive)
    mkpath(raw_net_path)
    brains = [h.bee_list[i].brain for i in 1:h.n_bees]

    serialize(string(raw_net_path, "epoch_", h.epoch_index, ".brains"), brains)
    return nothing
end

function run_regression(; n_bees, n_epochs, n_peaks, which_peak, trainsetsize, testsetsize)
    h = Hive(n_bees, n_epochs)
    sin_traindata = create_sin_dataset(n_peaks, which_peak, trainsetsize)
    sin_testdata = create_sin_dataset(n_peaks, which_peak, testsetsize)
    sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
    sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
    gillespie_regression!(h, trainloader=sin_trainloader, testloader=sin_testloader, n_epochs=n_epochs)
end

function run_regression_sbatch(trainsetsize, testsetsize)
    h = Hive(N_BEES, N_EPOCHS)
    sin_traindata = create_sin_dataset(5, 1, trainsetsize)
    sin_testdata = create_sin_dataset(5, 1, testsetsize)
    sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
    sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
    save_taskdata(RAW_TASKDATA_PATH, sin_traindata, sin_testdata)
    gillespie_regression!(h, trainloader=sin_trainloader, testloader=sin_testloader, n_epochs=N_EPOCHS, acc_atol=ACCURACY_ATOL, lambda_train=LAMBDA_TRAIN, lambda_interact=LAMBDA_INTERACT, n_steps_per_epoch=N_STEPS_PER_EPOCH)
end

function save_taskdata(raw_taskdata_path, traindata, testdata)
    mkpath(raw_taskdata_path)
    df_train_features = DataFrame(traindata[1], :auto)
    df_train_targets = DataFrame(traindata[2], :auto)
    df_test_features = DataFrame(testdata[1], :auto)
    df_test_targets = DataFrame(testdata[2], :auto)
    CSV.write(string(raw_taskdata_path, "train_features.csv"), df_train_features)
    CSV.write(string(raw_taskdata_path, "train_targets.csv"), df_train_targets)
    CSV.write(string(raw_taskdata_path, "test_features.csv"), df_test_features)
    CSV.write(string(raw_taskdata_path, "test_targets.csv"), df_test_targets)
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

