##
mo2 = build_cifar10_model()
for epoch in 1:100
    loss = train_model!(mo2, trainloader_cifar, learning_rate=0.0001)
    accuracy = calc_accuracy(mo2, testloader_cifar)
    println("Epoch = $(epoch) : Loss = $(loss) : Accuracy = $(accuracy)")
    total_loss += loss
end
println("total loss = $(total_loss)")
loss = train_model!(mo, trainloader)
calc_accuracy(mo, testloader_mnist)
calc_accuracy(mo, testloader_cifar)
calc_accuracy(mo, testloader_fashion)
calc_accuracy(mo, testloader_svhn)
testloader_cifar == testloader_fashion

##test logitcrossentropy
loss_fn(y_hat, y) = Flux.logitcrossentropy(y_hat, y)
loss_fn(y_model_mnist_train, y_mnist_train)

run_mnist_subset()



## test export_data
example1_nepoch = 1
loss_history = reshape([1,2,3,4],:,1)
n_ids=4
col_name = "loss_history"
accuracy_history = zeros(Float32, example1_nepoch)
export_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/test_export_data_function/"
filename = "dummy_data.csv"
mkpath(export_path)
epoch_ids = collect(1:example1_nepoch)
export_data(string(export_path, filename), loss_history, n_ids ,epoch_ids, col_name)


svx, svy = prepare_svhn2_dataset_greyscale(:train)
fashionx, fashiony  = prepare_fashionmnist_dataset_1_channel(:train)
mnistx, mnisty = prepare_mnist_dataset_1_channel(:train, subset=10000)
x_train, y_train = prepare_mnist_dataset_1_channel(:train)
x_test, y_test = prepare_mnist_dataset_1_channel(:test)
trainloader = Flux.DataLoader((x_train, y_train), batchsize=128, shuffle=true)
testloader = Flux.DataLoader((x_test, y_test), batchsize=128, shuffle=true)
test_hive = Hive(UInt16(5), UInt16(5))


function gillespie_testing(h::Hive; trainloader, testloader, n_epochs=DEFAULTS[:N_EPOCHS], lambda_Train=DEFAULTS[:LAMBDA_TRAIN], lambda_Interact=DEFAULTS[:LAMBDA_INTERACT], n_steps_per_epoch=DEFAULTS[:N_STEPS_PER_EPOCH])
    total_elapsed_time = 0.0
    gillespie_time = Float64(0.0)
    for bee in h.bee_list
        initial_accuracy = 0.01
        h.initial_accuracies_list[bee.id] = initial_accuracy
        h.current_accuracies_list[bee.id] = initial_accuracy
        bee.current_accuracy = initial_accuracy
    end
    h.current_accuracies_list[1] = 0.99
    h.bee_list[1].current_accuracy = 0.99
    for epoch in 1:n_epochs
        epoch_start_time = time()
        @info "Starting epoch $(epoch)"
        n_actions = 0
        n_train =0
        n_interact = 0
        n_dom_higher_acc = 0
        n_sub_higher_acc = 0
        while gillespie_time < epoch

            loop_start_time = time()
            a_train = lambda_Train * h.n_bees #propensity for training the neural networks. All networks (Bees) train with the same rate @lambda_Train
            K_matrix = compute_K_matrix(h.current_accuracies_list, lambda_Interact=lambda_Interact)
            a_interact = sum(K_matrix)

            total_propensity = a_train + a_interact

            d_t = rand(Exponential(1 / (n_steps_per_epoch * h.n_bees)))
            gillespie_time += d_t

            choose_action = rand() * total_propensity
            if choose_action < a_train

                selected_bee = h.bee_list[rand(1:h.n_bees)]
                loss = train_regression_model!(selected_bee.brain, trainloader, learning_rate=DEFAULTS[:LEARNING_RATE])

                h.loss_history[selected_bee.id, epoch] += loss
                current_accuracy = calc_accuracy(selected_bee.brain, testloader)
                h.current_accuracies_list[selected_bee.id] = current_accuracy
                selected_bee.current_accuracy = current_accuracy

                n_train +=1
                #println("n train: $(n_train)")
                #println("trained bee = $(selected_bee.id) : current accuracy = $(selected_bee.current_accuracy)")
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
                if sub_bee.id == 1
                    println("bee $(sub_bee.id) is being dominated")
                    println("bee $(dom_bee.id) is dominating")
                end

                #println("sub bee is $(sub_bee.id) : current accuracy=$(sub_bee.current_accuracy)")
                #println("dom bee is $(dom_bee.id) : current accuracy=$(dom_bee.current_accuracy)")

                n_interact+=1
                #println("n interact: $(n_interact)")
            end
            n_actions +=1
            epoch_loop_elapsed_time = time() - loop_start_time
            #@info "Epoch $(epoch) loop $(n_actions) completed" propensity_ratio=(a_train/a_interact) loop_time=epoch_loop_elapsed_time
        end
        h.accuracy_history[:, epoch] = h.current_accuracies_list[:, 1]
        elapsed_time = time() - epoch_start_time
        total_elapsed_time += elapsed_time
        @info "Epoch $(epoch) completed" epoch=epoch elapsed_time=elapsed_time gillespie_time=gillespie_time average_accuracy=mean(h.current_accuracies_list[:,1]) n_actions=n_actions n_train=n_train n_interact=n_interact dom_acc_higher=n_dom_higher_acc sub_acc_higher=n_sub_higher_acc
        @info "Memory usage $(Sys.total_memory()) bytes"
        save_nn_state(RAW_NET_PATH, h)
        h.epoch_index+=1
    end
    save_data(RAW_PATH, h, n_epochs)
    @info "Gillespie simulation is over. Data path: $(RAW_PATH)" total_elapsed_time=total_elapsed_time
end


gillespie_testing(test_hive, trainloader=trainloader, testloader=testloader, n_epochs=5, lambda_Train=0, lambda_Interact=50)

##test influence of changing one weight
model = build_model_4()
x_train, y_train = prepare_mnist_dataset_1_channel(:train)
x_test, y_test = prepare_mnist_dataset_1_channel(:test)
trainloader = Flux.DataLoader((x_train, y_train), batchsize=128, shuffle=true)
testloader = Flux.DataLoader((x_test, y_test), batchsize=128, shuffle=true)
train_model!(model, trainloader)
weight1 = deepcopy(model[1].weight[1,1,1,1])
weight2 = deepcopy(model[1].weight[1,1,1,2])
weight3 = deepcopy(model[4].weight[1,1,1,1])
weight4 = deepcopy(model[20].weight[1,1])
weight5 = deepcopy(model[10].weight[1,1,1,1])



current_acc = calc_accuracy(model, testloader)
model[1].weight[1,1,1,1] = weight1
model[1].weight[1,1,1,2] = weight2
model[4].weight[1,1,1,1] = weight3
model[20].weight[1,1] = weight4
model[10].weight[1,1,1,1] = 3



sin_traindata = create_sin_dataset(5, 1, 10000)
sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
sin_testdata = create_sin_dataset(5, 1, 1000)
sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
sin_model = build_model_sin()

init_acc = calc_regression_accuracy(sin_model, sin_testloader)
for i in 1:10
    train_regression_model!(sin_model, sin_trainloader)
end
new_acc = calc_regression_accuracy(sin_model, sin_testloader, atol=0.5)

diff = sum(abs.(vec(sin_traindata[2]) .- vec(sin_model(sin_traindata[1]))))


scatter(vec(sin_traindata[1]), vec(sin_traindata[2]))
y_pred = sin_model(sin_traindata[1])
scatter(vec(sin_traindata[1]), vec(y_pred))

result = isapprox.(vec(sin_traindata[2]), vec(y_pred), atol=0.005)

num_true = count(result)
num_false = count(!, result)

num_true/10000

result_test = isapprox.(vec(sin_testdata[2]), vec(sin_model(sin_testdata[2])), atol=0.5)
num_test_true = count(result_test)

correct = 0
total = 0
for (xb, yb) in sin_testloader
    test_pred = sin_model(xb)
    batch_result = isapprox.(vec(yb), vec(test_pred), atol=0.005)
    num_true_batch = count(batch_result)
    correct += num_true_batch
    total += length(yb)
end

test_sin_data = create_sin_dataset(3, 2, 3000)
sinx = vec(test_sin_data[1])
siny = vec(test_sin_data[2])
scatter(sinx, siny)


##
sin_hive = Hive_fixed_dims(UInt16(5), UInt16(50))
sin_traindata = create_sin_dataset(5, 1, 10000)
sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
sin_testdata = create_sin_dataset(5, 1, 1000)
sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
gillespie_