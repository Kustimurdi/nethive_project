
"""
the function @compute_K_matrix calculates the interaction propensities for all pairs of @Bee objects in accordance to the interaction rate stated in the wasp paper 
paramteres: 
- current_accuracies_list: a list that holds the current accuracies of the neural networks of the @Bee objects (the accuracy is a proxy for the queen gene expression)
- lambda_Interact: a parameter for the sigmoid function that gives the probability of bee_i being in a subdominant interaction with bee_j
"""
function compute_K_matrix(; queen_genes_list, lambda_interact)
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

function get_task_instance(symbol::Symbol)::Task
    if symbol == :regression
        return RegressionTask()
    elseif symbol == :linear_regression
        return LinearRegressionTask()
    elseif symbol == :none
        return NoTask()
    else
        error("Unsupported task symbol: $symbol")
    end
end

function get_qgm_instance(symbol::Symbol)::QueenGeneMethod
    if symbol == :from_accuracy
        return QueenGeneFromAccuracy()
    elseif symbol == :none
        return NoQueenGene()
    else
        error("Unsupported queen gene method symbol: $symbol")
    end
end


function gillespie_simulation!(h::Hive, h_paths::HivePaths)

    trainloader, testloader, train_data, test_data = create_dataset(hive.config.task_type, hive.config.task_config)
    save_taskdata(h_paths.raw_taskdata_path, train_data, test_data)

    queen_gene_method_struct = get_qgm_instance(h.config.queen_gene_method)
    task_type_struct = get_task_instance(h.config.task_type)

    total_elapsed_time = 0.0
    gillespie_time = Float64(0.0)

    #Initialize accuracies at the start of the simulation
    for bee in h.bee_list
        initial_accuracy = calc_accuracy(bee.brain, testloader, task_type_struct, acc_sigma=h.config.accuracy_sigma)
        h.initial_accuracies_list[bee.id] = initial_accuracy
    end
    save_nn_state(RAW_NET_PATH, h)

    queen_genes_vector = [bee.queen_gene for bee in hive.bee_list]

    for epoch in 1:h.config.n_epochs
        epoch_start_time = time()
        @info "Starting epoch $(epoch)"

        n_actions = 0
        n_train =0
        n_interact = 0

        while gillespie_time < epoch
            loop_start_time = time()

            # Calculate the interaction propensity for the current epoch
            a_train = h.config.lambda_train * h.config.n_bees 
            K_matrix = compute_K_matrix(queen_genes_list=queen_genes_vector, lambda_interact=h.config.lambda_interact)
            a_interact = sum(K_matrix)

            total_propensity = a_train + a_interact
            d_t = rand(Exponential(1 / (h.config.n_steps_per_epoch * h.config.n_bees)))
            gillespie_time += d_t

            choose_action = rand() * total_propensity
            if choose_action < a_train
                # Train a bee
                selected_bee = h.bee_list[rand(1:h.config.n_bees)]
                loss = train_model!(selected_bee.brain, trainloader, task_type_struct, learning_rate=h.config.learning_rate)
                h.loss_history[selected_bee.id, epoch] += loss

                # Calculate the new queen gene
                new_queen_gene = compute_queen_gene(selected_bee, testloader, task_type_struct, queen_gene_method_struct)
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
                punish_regression_model!(sub_bee.brain, trainloader, h.config.punish_rate, task_type_struct)
                loss = calc_regression_loss(sub_bee.brain, testloader)
                new_queen_gene = compute_queen_gene(selected_bee, testloader, task_type_struct, queen_gene_method_struct)
                if h.queen_gene_method isa QueenGeneIncremental
                    new_queen_gene = sub_bee.queen_gene - h.config.queen_gene_method.increment_value
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
            h.accuracy_history[bee.id, epoch] = calc_accuracy(bee.brain, testloader, h.config.task_type, acc_sigma=h.config.accuracy_sigma)
        end

        h.propensity_ratio_history[epoch] = a_train / a_interact

        # Save the queen genes
        h.queen_genes_history[:, epoch] = queen_genes_vector
        elapsed_time = time() - epoch_start_time
        total_elapsed_time += elapsed_time

        @info "Epoch $(epoch) completed" epoch=epoch elapsed_time=elapsed_time gillespie_time=gillespie_time average_accuracy=mean(h.current_accuracies_list[:,1]) n_actions=n_actions n_train=n_train n_interact=n_interact 
        save_nn_state(h_paths.raw_net_path, h)
    end
    
    # Save the task data
    #save_taskdata(RAW_PATH, h)
    @info "Gillespie simulation is over. " total_elapsed_time=total_elapsed_time
end

function save_data(raw_path::String, h::Hive)
    mkpath(raw_path) 
    epoch_ids = collect(1:h.n_epochs)

    data_fields = Dict(
        "accuracy" => h.accuracy_history,
        "loss" => h.loss_history,
        "queen_gene" => h.queen_genes_history,
        "train_count" => h.n_train_history,
        "subdominant_count" => h.n_subdominant_history,
        "dominant_count" => h.n_dominant_history
    )

    for (name, data) in data_fields
        export_data(joinpath(raw_path, name * ".csv"), data, h.n_bees, epoch_ids, name)
    end

    return 0
end

function save_nn_state(raw_net_path::String, h::Hive, filename_prefix::String = "epoch_")
    mkpath(raw_net_path)

    brain_states = Dict(
        "brains" => [bee.brain for bee in h.bee_list],
        "observation_histories" => [bee.observation_history for bee in h.bee_list]
    )

    for (name, data) in brain_states
        serialize(joinpath(raw_net_path, "$(filename_prefix)$(h.epoch_index).$(name)"), data)
    end

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

function save_simulation_params(config::HiveConfig, raw_path::String)
    mkpath(raw_path)
    println("Data path: ", raw_path)

    params = Dict(Symbol(field) => config[field] for field in fieldnames(HiveConfig))
    task_params = Dict(Symbol(field) => config.task_config[field] for field in fieldnames(TaskConfig))
    params = merge(params, task_params)

    df = DataFrame(params, :auto)
    df[!, :id] = 1:nrow(df)

    # Add metadata
    metadata = Dict(
        "git_branch" => get_git_branch(),
        "git_commit_id" => get_git_commit()
    )

    for (key, value) in metadata
        push!(df, (key, value))
    end

    CSV.write(joinpath(raw_path, "parameters.csv"), df, writeheader=true)
    return 0
end
