"""
the function @compute_K_matrix calculates the interaction propensities for all pairs of @Bee objects in accordance to the interaction rate stated in the wasp paper 
paramteres: 
- current_accuracies_list: a list that holds the current accuracies of the neural networks of the @Bee objects (the accuracy is a proxy for the queen gene expression)
- lambda_Interact: the exponent prefactor of the sigmoid function of the interaction rate
"""
function compute_K_matrix(; queen_genes_list, lambda_interact)
    K_matrix = K_func.(queen_genes_list, queen_genes_list', lambda_interact)
    K_matrix[diagind(K_matrix)] .= 0
    return K_matrix 
end

function K_func(r_i, r_j, lambda) 
    #return r_i*r_j*Flux.sigmoid(-lambda*(r_i - r_j))
    result = r_i*r_j*Flux.sigmoid(-lambda*(r_i - r_j))
    return result
end

function choose_interaction(n_bees, K_matrix)
    choose_interaction = rand() * sum(K_matrix)
    cumulative_sum = 0
    for i in 1:n_bees
        for j in 1:n_bees
            cumulative_sum += K_matrix[i, j]
            if cumulative_sum >= choose_interaction
                return (i, j)
            end
        end 
    end 
end

function gillespie_simulation!(h::Hive, h_paths::HivePaths; save_data::Bool=true)

    queen_gene_method_struct = get_qgm_instance(h.config.queen_gene_method)
    task_type_struct = get_task_instance(h.config.task_type)
    println("task_type_struct: ", task_type_struct)
    println("queen_gene_method_struct: ", queen_gene_method_struct)

    trainloader, testloader, train_data, test_data = create_dataset(task_type_struct, h.config.task_config)

    total_elapsed_time = 0.0
    gillespie_time = Float64(0.0)

    n_actions = 0
    n_train = 0
    n_interact = 0
    propensity_ratio = 0.0
    a_interact = 0.0

    K_matrix = nothing

    #Initial values
    for bee in h.bee_list
        initial_accuracy = calc_accuracy(bee.brain, testloader, task_type_struct, acc_sigma=h.config.accuracy_sigma)
        h.accuracy_history[bee.id, 1] = initial_accuracy
        bee.queen_gene = compute_queen_gene(bee, queen_gene_method_struct, testloader, task_type_struct, h.config.accuracy_sigma)
        #h.queen_genes_history[bee.id, 1] = bee.queen_gene
    end

    if save_data && (h.config.save_nn_epochs>0)
        save_nn_state(h_paths.raw_net_path, h)
    end

    queen_genes_vector = [bee.queen_gene for bee in h.bee_list]
    println("queen_genes_vector: ", queen_genes_vector)

    for epoch in 1:h.config.n_epochs
        epoch_start_time = time()
        @info "Starting epoch $(epoch)"

        while gillespie_time < epoch
            loop_start_time = time()

            # Calculate the interaction propensity for the current epoch
            a_train = h.config.training_propensity * h.config.n_bees 
            K_matrix = compute_K_matrix(queen_genes_list=queen_genes_vector, lambda_interact=h.config.lambda_interact)
            a_interact = sum(K_matrix)

            total_propensity = a_train + a_interact
            d_t = rand(Exponential(1 / (h.config.n_steps_per_epoch * h.config.n_bees)))
            gillespie_time += d_t

            choose_action = rand() * total_propensity

            #@info "debug info" a_train=a_train a_interact=a_interact total_propensity=total_propensity choose_action=choose_action lambda_interact=h.config.lambda_interact
            if choose_action > total_propensity
                println("choose_action > total_propensity")
                break
            end
            if choose_action < a_train
                # Train a bee
                selected_bee = h.bee_list[rand(1:h.config.n_bees) ]
                loss = train_model!(selected_bee.brain, trainloader, task_type_struct, learning_rate=h.config.learning_rate)
                h.loss_history[selected_bee.id, epoch] += loss

                # Calculate the new queen gene
                new_queen_gene = compute_queen_gene(selected_bee, queen_gene_method_struct, testloader, task_type_struct, h.config.accuracy_sigma)
                if queen_gene_method_struct isa QueenGeneIncremental
                    new_queen_gene = selected_bee.queen_gene + queen_gene_method_struct.increment_value
                end
                if new_queen_gene < 0
                    new_queen_gene = 0
                end
                selected_bee.queen_gene = new_queen_gene
                h.n_train_history[selected_bee.id, epoch] += 1

                n_train +=1
            else
                sub_bee_id, dom_bee_id = choose_interaction(h.config.n_bees, K_matrix)
                sub_bee, dom_bee = h.bee_list[sub_bee_id], h.bee_list[dom_bee_id]

                # Apply the interaction and update the queen gene
                #punish_model!(sub_bee.brain, trainloader, task_type_struct, punish_rate=h.config.punish_rate)
                punish_model_resetting!(sub_bee, h.config.task_config)
                new_queen_gene = compute_queen_gene(sub_bee, queen_gene_method_struct, testloader, task_type_struct, h.config.accuracy_sigma)
                if queen_gene_method_struct isa QueenGeneIncremental
                    new_queen_gene = sub_bee.queen_gene - queen_gene_method_struct.decrement_value
                end
                if new_queen_gene < 0
                    new_queen_gene = 0
                end
                old_queen_gene = sub_bee.queen_gene
                sub_bee.queen_gene = new_queen_gene

                push!(h.interaction_log[epoch], (sub_bee_id, dom_bee_id))
                #h.n_subdominant_history[sub_bee.id, epoch] += 1
                #h.n_dominant_history[dom_bee.id, epoch] +=1

                n_interact+=1
            end

            n_actions +=1
            propensity_ratio = a_interact / a_train
            epoch_loop_elapsed_time = time() - loop_start_time

            queen_genes_vector = [bee.queen_gene for bee in h.bee_list]

            #@info "Epoch $(epoch) loop $(n_actions) completed" propensity_ratio=(a_train/a_interact) loop_time=epoch_loop_elapsed_time a_train=a_train a_interact=a_interact
        end
        h.epoch_index+=1

        """
        if h.epoch_index%3000==0
            println("task type struct: ", task_type_struct)
            first_bee = h.bee_list[1] 
            punish_model_resetting!(first_bee, task_type_struct)
            h.n_subdominant_history[1, epoch] =+1
        end
        """

        # Calculate the accuracy for each bee
        for bee in h.bee_list
            h.accuracy_history[bee.id, epoch+1] = calc_accuracy(bee.brain, testloader, task_type_struct, acc_sigma=h.config.accuracy_sigma)
            h.dominant_rate_history[bee.id, epoch] = sum(K_matrix[:, bee.id])
            h.subdominant_rate_history[bee.id, epoch] = sum(K_matrix[bee.id, :])
            #if h.n_train_history[bee.id, epoch] != 0
            #    h.loss_history[bee.id, epoch] = h.loss_history[bee.id, epoch] / h.n_train_history[bee.id, epoch]
            #end
        end

        h.loss_history[:, epoch] .= ifelse.(h.n_train_history[:, epoch] .!= 0, h.loss_history[:, epoch] ./ h.n_train_history[:, epoch], h.loss_history[:, epoch])
        h.propensity_ratio_history[epoch] = propensity_ratio

        # Save the queen genes
        h.queen_genes_history[:, epoch+1] = queen_genes_vector
        elapsed_time = time() - epoch_start_time
        total_elapsed_time += elapsed_time

        @info "Epoch $(epoch) completed" epoch=epoch elapsed_time=elapsed_time gillespie_time=gillespie_time average_accuracy=mean(h.accuracy_history[:,epoch]) n_actions=n_actions n_train=n_train n_interact=n_interact propensity_ratio=propensity_ratio
        if save_data && (h.config.save_nn_epochs>0) 
            if h.epoch_index%h.config.save_nn_epochs == 0
                save_nn_state(h_paths.raw_net_path, h)
            end
        end
    end
    
    # Save data
    if save_data
        save_taskdata(h_paths.raw_taskdata_path, train_data, test_data)
        save_simulation_params_wide(h.config, h_paths.raw_path)
        save_results(joinpath(h_paths.raw_path, "raw"), h)
    end
    @info "Gillespie simulation is over. " total_elapsed_time=total_elapsed_time n_actions=n_actions n_train=n_train n_interact=n_interact
    println("queen gene struct: ", queen_gene_method_struct)
    return h, trainloader, testloader, train_data, test_data
end

function save_results(raw_path::String, h::Hive)
    mkpath(raw_path) 

    data_fields = Dict(
        "accuracy" => h.accuracy_history,
        "loss" => h.loss_history,
        "train_count" => h.n_train_history,
        #"queen_gene" => h.queen_genes_history,
        #"subdominant_count" => h.n_subdominant_history,
        #"dominant_count" => h.n_dominant_history,
        "subdominant_rates" => h.subdominant_rate_history,
        "dominant_rates" => h.dominant_rate_history
    )

    for (name, data) in data_fields
        export_data(joinpath(raw_path, name * ".csv"), data, name)
        #export_data(joinpath(raw_path, name * ".csv"), data, h.n_bees, epoch_ids, name)
    end
    vector_to_dataframe(joinpath(raw_path, "propensity_ratio.csv"), h.propensity_ratio_history, :propensity_ratio)

    export_interaction_log(raw_path, h.interaction_log)

    return 0
end

function export_interaction_log(raw_path::String, interaction_log)
    rows = []
    for (epoch, events) in enumerate(interaction_log)
        for (sub, dom) in events
            push!(rows, (epoch=epoch, dominant=dom, subdominant=sub))
        end
    end
    df = DataFrame(rows)
    CSV.write(joinpath(raw_path, "interaction_log.csv"), df)
end

function save_nn_state(raw_net_path::String, h::Hive, filename_prefix::String = "epoch_")
    mkpath(raw_net_path)

    brain_states = Dict(
        "brains" => [bee.brain for bee in h.bee_list]
        #"observation_histories" => [bee.observation_history for bee in h.bee_list]
    )

    for (name, data) in brain_states
        serialize(joinpath(raw_net_path, "$(filename_prefix)$(h.epoch_index).$(name)"), data)
    end

    return 0
end

function save_taskdata(raw_taskdata_path, traindata, testdata)
    if isnothing(traindata) || isnothing(testdata)
        println("No task data to save.")
        return
    end
    mkpath(raw_taskdata_path)
    df_train_features = DataFrame(traindata[1], :auto)
    df_train_targets = DataFrame(traindata[2], :auto)
    df_test_features = DataFrame(testdata[1], :auto)
    df_test_targets = DataFrame(testdata[2], :auto)
    CSV.write(joinpath(raw_taskdata_path, "train_features.csv"), df_train_features)
    CSV.write(joinpath(raw_taskdata_path, "train_targets.csv"), df_train_targets)
    CSV.write(joinpath(raw_taskdata_path, "test_features.csv"), df_test_features)
    CSV.write(joinpath(raw_taskdata_path, "test_targets.csv"), df_test_targets)
end

"""
function save_simulation_params(config::HiveConfig, raw_path::String)
    mkpath(raw_path)
    println("Data path: ", raw_path)

    params = Dict(Symbol(field) => getfield(config, field) for field in fieldnames(HiveConfig))
    task_params = Dict(Symbol(field) => getfield(config.task_config, field) for field in fieldnames(typeof(config.task_config)))
    params = merge(params, task_params)


    # Add metadata
    metadata = Dict(
        "git_branch" => get_git_branch(),
        "git_commit_id" => get_git_commit()
    )
    
    params = merge(params, metadata)

    #for (key, value) in metadata
    #    push!(df, (key, value))
    #end

    #df = DataFrame(params, :auto)
    df = DataFrame([(k, v) for (k, v) in params], [:parameter, :value])

    df[!, :id] = 1:nrow(df)

    CSV.write(joinpath(raw_path, "parameters.csv"), df, writeheader=true)
    return 0
end
"""

function save_simulation_params(config::HiveConfig, raw_path::String)
    mkpath(raw_path)
    println("Data path: ", raw_path)

    # Collect fields from the main config and task config
    params = Dict(string(field) => getfield(config, field) for field in fieldnames(HiveConfig))
    task_params = Dict(string(field) => getfield(config.task_config, field) for field in fieldnames(typeof(config.task_config)))
    all_params = merge(params, task_params)

    # Add metadata (as Strings for clarity)
    metadata = Dict(
        "git_branch" => string(get_git_branch()),
        "git_commit_id" => string(get_git_commit())
    )
    all_params = merge(all_params, metadata)
    pop!(all_params, "task_config", nothing)

    # Prepare rows for the DataFrame
    rows = [(i, k, string(v)) for (i, (k, v)) in enumerate(all_params)]

    # Create DataFrame with correct column names
    df = DataFrame(rows, [:id, :parameter, :value])

    # Save to CSV
    CSV.write(joinpath(raw_path, "parameters.csv"), df, writeheader=true)
    return 0
end

function save_simulation_params_wide(config::HiveConfig, raw_path::String)
    mkpath(raw_path)
    println("Data path: ", raw_path)

    # Collect fields from the main config and task config
    params = Dict(string(field) => getfield(config, field) for field in fieldnames(HiveConfig))
    task_params = Dict(string(field) => getfield(config.task_config, field) for field in fieldnames(typeof(config.task_config)))
    all_params = merge(params, task_params)

    # Add metadata (as Strings for clarity)
    metadata = Dict(
        "git_branch" => string(get_git_branch()),
        "git_commit_id" => string(get_git_commit())
    )
    all_params = merge(all_params, metadata)
    pop!(all_params, "task_config", nothing)

    # Create DataFrame in wide format (each parameter is a column)
    df = DataFrame(all_params; copycols=true)

    # Save to CSV
    CSV.write(joinpath(raw_path, "parameters.csv"), df, writeheader=true)
    return 0
end
