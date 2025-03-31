function gillespie_test!(h::Hive; trainloader, testloader, n_epochs=DEFAULTS[:N_EPOCHS], acc_atol=DEFAULTS[:ACCURACY_ATOL], lambda_Train=DEFAULTS[:LAMBDA_TRAIN], lambda_Interact=DEFAULTS[:LAMBDA_INTERACT], n_steps_per_epoch=DEFAULTS[:N_STEPS_PER_EPOCH])
    total_elapsed_time = 0.0
    gillespie_time = Float64(0.0)
    for bee in h.bee_list
        initial_accuracy = calc_regression_accuracy(bee.brain, testloader, atol=acc_atol)
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
        list_no_trained=fill(0, h.n_bees)
        while gillespie_time < epoch

            loop_start_time = time()
            a_train = lambda_Train * h.n_bees #propensity for training the neural networks. All networks (Bees) train with the same rate @lambda_Train
            #K_matrix = compute_K_matrix(h.current_accuracies_list, lambda_Interact=lambda_Interact)
            #a_interact = sum(K_matrix)
            a_interact = 0.0

            total_propensity = a_train + a_interact

            d_t = rand(Exponential(1 / (n_steps_per_epoch * h.n_bees)))
            gillespie_time += d_t

            choose_action = rand() * total_propensity
            if choose_action < a_train

                selected_bee = h.bee_list[rand(1:h.n_bees)]
                list_no_trained[selected_bee.id] += 1
                if selected_bee.id == 1
                    println("training cursed bee 1")
                end 
                loss = train_regression_model!(selected_bee.brain, trainloader, learning_rate=DEFAULTS[:LEARNING_RATE])

                h.loss_history[selected_bee.id, epoch] += loss
                current_accuracy = calc_regression_accuracy(selected_bee.brain, testloader, atol=acc_atol)
                h.current_accuracies_list[selected_bee.id] = current_accuracy
                if selected_bee.id == 1
                    println("difference in accs: $(current_accuracy - selected_bee.current_accuracy)")
                end
                selected_bee.current_accuracy = current_accuracy

                n_train +=1
            else
                """
                #!!!!!!!!!!!!hier weiter implimentieren
                sub_bee, dom_bee = choose_interaction(h, a_interact, K_matrix)

                prefactor = 0.001
                #punish_regression_model!(sub_bee.brain, trainloader, prefactor)
                new_accuracy = calc_regression_accuracy(sub_bee.brain, testloader, atol=acc_atol)
                println("bee id = $(sub_bee.id) : old acc = $(sub_bee.current_accuracy) : new acc = $(new_accuracy)")

                h. current_accuracies_list[sub_bee.id] = new_accuracy
                sub_bee.current_accuracy = new_accuracy

                h.n_subdom_interactions_history[sub_bee.id, epoch] += 1
                h.n_dom_interactions_history[dom_bee.id, epoch] +=1

                n_interact+=1
                println("n interact: $(n_interact)")
                """
            end
            n_actions +=1
            epoch_loop_elapsed_time = time() - loop_start_time
            @info "Epoch $(epoch) loop $(n_actions) completed" propensity_ratio=(a_train/a_interact) loop_time=epoch_loop_elapsed_time a_train=a_train a_interact=a_interact
        end
        println("trained list = $(list_no_trained)")
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


function run_testing(; n_bees, n_epochs, n_peaks, which_peak, trainsetsize, testsetsize)
    h = Hive(n_bees, n_epochs, brain_constructor=build_model_sin_leaky)
    sin_traindata = create_sin_dataset(n_peaks, which_peak, trainsetsize)
    sin_testdata = create_sin_dataset(n_peaks, which_peak, testsetsize)
    sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
    sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
    gillespie_test_first_bee!(h, trainloader=sin_trainloader, testloader=sin_testloader, n_epochs=n_epochs)
end

function run_testing_sbatch(trainsetsize, testsetsize)
    h = Hive(N_BEES, N_EPOCHS)
    sin_traindata = create_sin_dataset(5, 1, trainsetsize)
    sin_testdata = create_sin_dataset(5, 1, testsetsize)
    sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
    sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
    gillespie_test!(h, trainloader=sin_trainloader, testloader=sin_testloader, n_epochs=N_EPOCHS, acc_atol=ACCURACY_ATOL, lambda_Train=LAMBDA_TRAIN, lambda_Interact=LAMBDA_INTERACT, n_steps_per_epoch=N_STEPS_PER_EPOCH)
end

function straightuptrain!(h::Hive; trainloader, testloader, n_epochs, acc_atol)
    total_elapsed_time = 0.0
    for bee in h.bee_list
        initial_accuracy = calc_regression_accuracy(bee.brain, testloader, atol=acc_atol)
        h.initial_accuracies_list[bee.id] = initial_accuracy
        h.current_accuracies_list[bee.id] = initial_accuracy
        bee.current_accuracy = initial_accuracy
    end
    for epoch in 1:n_epochs
        epoch_start_time = time()
        for selected_bee in h.bee_list
            loss = train_regression_model!(selected_bee.brain, trainloader, learning_rate=DEFAULTS[:LEARNING_RATE])

            h.loss_history[selected_bee.id, epoch] += loss
            current_accuracy = calc_regression_accuracy(selected_bee.brain, testloader, atol=acc_atol)
            h.current_accuracies_list[selected_bee.id] = current_accuracy
            selected_bee.current_accuracy = current_accuracy
        end
        h.accuracy_history[:, epoch] = h.current_accuracies_list[:, 1]
        elapsed_time = time() - epoch_start_time
        total_elapsed_time += elapsed_time
        @info "Epoch $(epoch) completed" epoch=epoch elapsed_time=elapsed_time average_accuracy=mean(h.current_accuracies_list[:,1]) current_accuracies=h.current_accuracies_list 
        @info "Memory usage $(Sys.total_memory()) bytes"
        save_nn_state(RAW_NET_PATH, h)
    end
    save_data(RAW_PATH, h, n_epochs)
    @info "straightuptrain simulation is over. Data path: $(RAW_PATH)" total_elapsed_time=total_elapsed_time
end

function run_straightuptrain(trainsetsize, testsetsize)
    h = Hive(N_BEES, N_EPOCHS)
    sin_traindata = create_sin_dataset(5, 1, trainsetsize)
    sin_testdata = create_sin_dataset(5, 1, testsetsize)
    sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
    sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
    straightuptrain!(h, trainloader=sin_trainloader, testloader=sin_testloader, n_epochs=N_EPOCHS, acc_atol=ACCURACY_ATOL)
end

trainsetsize=10000
testsetsize=1000
h = Hive(UInt16(5), UInt16(380))
sin_traindata = create_sin_dataset(5, 3, trainsetsize)
sin_testdata = create_sin_dataset(5, 3, testsetsize)
sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
straightuptrain!(h, trainloader=sin_trainloader, testloader=sin_testloader, n_epochs=UInt16(380), acc_atol=ACCURACY_ATOL)
gillespie_regression!(h, trainloader=sin_trainloader, testloader=sin_testloader, n_epochs=UInt(380))

plot_hive_predictions(h, sin_traindata[1])

plot_bee_prediction(h.bee_list[1].brain, sin_traindata[1])

plot_hive_history(h.accuracy_history)

plot_bee_history(h.accuracy_history, 4)

current_acc = calc_regression_accuracy(h.bee_list[1].brain, sin_testloader, atol=DEFAULTS[:ACCURACY_ATOL])

for i in 1:1
    loss = punish_regression_model!(h.bee_list[1].brain, sin_trainloader, 0.0000001)
    new_acc = calc_regression_accuracy(h.bee_list[1].brain, sin_testloader, atol=DEFAULTS[:ACCURACY_ATOL])
    println("epoch = $(i) : new acc = $(new_acc) : loss = $(loss)")
end

y_bee1_train = h.bee_list[1].brain(sin_traindata[1])
y_bee2_train = h.bee_list[2].brain(sin_traindata[1])
y_bee3_train = h.bee_list[3].brain(sin_traindata[1])
y_bee4_train = h.bee_list[4].brain(sin_traindata[1])


y_bee1_test = h.bee_list[1].brain(sin_testdata[1])
y_bee2_test = h.bee_list[2].brain(sin_testdata[1])
y_bee3_test = h.bee_list[3].brain(sin_testdata[1])
y_bee4_test = h.bee_list[4].brain(sin_testdata[1])

# Create a figure and axis
fig = Figure()
ax = Makie.Axis(fig[1, 1], title="Sine Wave", xlabel="x", ylabel="sin(x)")

# Plot the data
Makie.scatter!(ax, vec(sin_traindata[1]), vec(y_bee1_train), color=:blue, markersize=4)
Makie.scatter!(ax, vec(sin_traindata[1]), vec(y_bee2_train), color=:red, markersize=4)
Makie.scatter!(ax, vec(sin_traindata[1]), vec(y_bee3_train), color=:green, markersize=4)
Makie.scatter!(ax, vec(sin_traindata[1]), vec(y_bee4_train), color=:pink, markersize=4)


Makie.scatter!(ax, vec(sin_testdata[1]), vec(y_bee1_test), color=:blue, markersize=4)
Makie.scatter!(ax, vec(sin_testdata[1]), vec(y_bee2_test), color=:red, markersize=4)
Makie.scatter!(ax, vec(sin_testdata[1]), vec(y_bee3_test), color=:green, markersize=4)
Makie.scatter!(ax, vec(sin_testdata[1]), vec(y_bee4_test), color=:pink, markersize=4)

# Display the plot
fig

y_bee1


init_acc = calc_regression_accuracy(h.bee_list[3].brain, sin_testloader, atol=DEFAULTS[:ACCURACY_ATOL])
for i in 1:50
    loss = train_regression_model!(h.bee_list[3].brain, sin_trainloader)
    println(loss)
end
new_acc = calc_regression_accuracy(h.bee_list[3].brain, sin_testloader, atol=0.01)
loss_fn(x, y) = Flux.Losses.mse(h.bee_list[3].brain(x), y)
loss_fn(sin_traindata[1], sin_traindata[2])

fig = Figure()
ax = Makie.Axis(fig[1,1])
