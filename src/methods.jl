
"""
the function @compute_K_matrix calculates the interaction propensities for all pairs of @Bee objects in accordance to the interaction rate stated in the wasp paper 
paramteres: 
- current_accuracies_list: a list that holds the current accuracies of the neural networks of the @Bee objects (the accuracy is a proxy for the queen gene expression)
- lambda_Interact: a parameter for the sigmoid function that gives the probability of bee_i being in a subdominant interaction with bee_j
"""
function punish_model!(model::Flux.Chain, dataloader, punish_rate, task::Task)
    loss_fn(x, y) = compute_task_loss(model, x, y, task)
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

function compute_K_matrix(queen_genes_list; lambda_interact=DEFAULTS[LAMBDA_INTERACT])
    K_matrix = K_func.(queen_genes_list, queen_genes_list', lambda_interact)
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


function gillespie_regression!(h::Hive; 
                                trainloader, 
                                testloader, 
                                learning_rate=DEFAULTS[:LEARNING_RATE], 
                                punish_rate=DEFAULTS[:PUNISH_RATE], 
                                acc_sigma=DEFAULTS[:ACCURACY_ATOL], 
                                lambda_train=DEFAULTS[:LAMBDA_TRAIN], 
                                lambda_interact=DEFAULTS[:LAMBDA_INTERACT], 
                                n_steps_per_epoch=DEFAULTS[:N_STEPS_PER_EPOCH])

    total_elapsed_time = 0.0
    gillespie_time = Float64(0.0)

    #Initialize accuracies at the start of the simulation
    for bee in h.bee_list
        initial_accuracy = calc_accuracy(bee.brain, testloader, h.task_type, acc_sigma=acc_sigma)
        h.initial_accuracies_list[bee.id] = initial_accuracy
    end
    save_nn_state(RAW_NET_PATH, h)

    queen_genes_vector = [bee.queen_gene for bee in hive.bee_list]

    for epoch in 1:h.n_epochs
        epoch_start_time = time()
        @info "Starting epoch $(epoch)"

        n_actions = 0
        n_train =0
        n_interact = 0

        while gillespie_time < epoch
            loop_start_time = time()

            # Calculate the interaction propensity for the current epoch
            a_train = lambda_train * h.n_bees 
            K_matrix = compute_K_matrix(queen_genes_vector, lambda_interact=lambda_interact)
            a_interact = sum(K_matrix)

            total_propensity = a_train + a_interact
            d_t = rand(Exponential(1 / (n_steps_per_epoch * h.n_bees)))
            gillespie_time += d_t

            choose_action = rand() * total_propensity
            if choose_action < a_train
                # Train a bee
                selected_bee = h.bee_list[rand(1:h.n_bees)]
                loss = train_model!(selected_bee.brain, trainloader, h.task_type, learning_rate=learning_rate)
                h.loss_history[selected_bee.id, epoch] += loss

                # Calculate the new queen gene
                new_queen_gene = compute_queen_gene(selected_bee, testloader, h.task_type, h.queen_gene_method)
                selected_bee.queen_gene = new_queen_gene

                h.n_train_history[selected_bee.id, epoch] += 1

                n_train +=1
                println("n train: $(n_train)")
                println("n train: $(sum(h.n_train_history[:, epoch]))")
                println("trained bee = $(selected_bee.id) : current queen gene = $(selected_bee.queen_gene)")
            else
                # Interact with another bee
                sub_bee, dom_bee = choose_interaction(h, a_interact, K_matrix)

                # Apply the interaction and update the queen gene
                punish_regression_model!(sub_bee.brain, trainloader, punish_rate, h.task_type)
                loss = calc_regression_loss(sub_bee.brain, testloader)
                new_queen_gene = compute_queen_gene(selected_bee, testloader, h.task_type, h.queen_gene_method)
                if h.queen_gene_method == QueenGeneIncremental()
                    new_queen_gene = sub_bee.queen_gene - h.queen_gene_method.increment_value
                end
                sub_bee.queen_gene = new_queen_gene

                h.n_subdominant_history[sub_bee.id, epoch] += 1
                h.n_dominant_history[dom_bee.id, epoch] +=1

                n_interact+=1
                println("n interact: $(n_interact)")
                println("n interact: $(sum(h.n_dom))")
                println("bee id = $(sub_bee.id) : old queen gene = $(sub_bee.queen_gene) : new queen gene = $(new_queen_gene)")
            end

            n_actions +=1
            epoch_loop_elapsed_time = time() - loop_start_time

            queen_genes_vector = [bee.queen_gene for bee in hive.bee_list]

            @info "Epoch $(epoch) loop $(n_actions) completed" propensity_ratio=(a_train/a_interact) loop_time=epoch_loop_elapsed_time a_train=a_train a_interact=a_interact
        end
        h.epoch_index+=1

        # Calculate the accuracy for each bee
        for bee in h.bee_list
            h.accuracy_history[bee.id, epoch] = calc_accuracy(bee.brain, testloader, h.task_type, acc_sigma=acc_sigma)
        end

        h.propensity_ratio_history[epoch] = a_train / a_interact

        # Save the queen genes
        h.queen_genes_history[:, epoch] = queen_genes_vector
        elapsed_time = time() - epoch_start_time
        total_elapsed_time += elapsed_time

        @info "Epoch $(epoch) completed" epoch=epoch elapsed_time=elapsed_time gillespie_time=gillespie_time average_accuracy=mean(h.current_accuracies_list[:,1]) n_actions=n_actions n_train=n_train n_interact=n_interact 
        save_nn_state(RAW_NET_PATH, h)
    end
    
    # Save the final data
    save_data(RAW_PATH, h, n_epochs)
    @info "Gillespie simulation is over. Data path: $(RAW_PATH)" total_elapsed_time=total_elapsed_time
end

function save_data(raw_path::String, h::Hive, n_epochs=DEFAULTS[:N_EPOCHS])
    mkpath(raw_path) 
    #epoch_ids = collect(1:h.epoch_index) 
    epoch_ids = collect(1:n_epochs)
    export_data(string(raw_path, "/accuracy_history", ".csv"), h.accuracy_history, h.n_bees, epoch_ids, "accuracy")
    export_data(string(raw_path, "/loss_history", ".csv"), h.loss_history, h.n_bees, epoch_ids, "loss")
    export_data(string(raw_path, "/queen_genes_history", ".csv"), h.queen_genes_history, h.n_bees, epoch_ids, "queen_gene")
    export_data(string(raw_path, "/train_history", ".csv"), h.n_train_history, h.n_bees, epoch_ids, "train_count")
    export_data(string(raw_path, "/subdominant_history", ".csv"), h.n_subdominant_history, h.n_bees, epoch_ids, "subdominant_count")
    export_data(string(raw_path, "/dominant_history", ".csv"), h.n_dominant_history, h.n_bees, epoch_ids, "dominant_count")
    return 0
end

function save_nn_state(raw_net_path::String, h::Hive)
    mkpath(raw_net_path)
    brains = [h.bee_list[i].brain for i in 1:h.n_bees]
    serialize(string(raw_net_path, "epoch_", h.epoch_index, ".brains"), brains)
    return 0
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

