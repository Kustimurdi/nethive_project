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
function calc_accuracy(model, dataloader, n_labels, num_batches::Int=typemax(Int))
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
the function @choose_partner_extrinsicly calculates the probability of interacting in given @epoch for every @bee in the Hive @h in accordance of the accuracy of the @bee during given epoch.
The accuracy thus has to be already calculated and stored in the accuracy_history
Depending on the argument for type the probability is either directly equal to the accuracy or the ratio of the accuracy and the sum of the accuracies
"""
function choose_partner!(h::Hive, epoch::UInt16, type::String="intrinsic")
    for i in 1:h.n_bees
        println("looking at bee number $(i)")
        if h.bee_list[i].id != i
            println("for loop index does not equal bee id")
        end
        if h.interaction_partner_history[i, epoch] != -1
            println("was already altered")
            continue
        end
        if type == "intrinsic"
            interaction_probability = h.accuracy_history[i, epoch]
        elseif type == "extrinsic"
            interaction_probability = h.accuracy_history[i, epoch]/epoch_accuracy_sum
        else
            throw(ArgumentError("Invalid Argument provided"))
        end
        println("Interaction probability: $(interaction_probability)")
        if rand() < interaction_probability #bee of bee_id i is interacting
            println("chose yes")
            interaction_partner = rand(1:(h.n_bees - 1)) #interaction partner is chosen
            if interaction_partner >= i
                interaction_partner+=1
            end
            if (h.interaction_partner_history[interaction_partner, epoch] == -1) || (h.interaction_partner_history[interaction_partner, epoch] == 0)
                println("partner has no partner yet")
                h.interaction_partner_history[i, epoch] = interaction_partner
                h.interaction_partner_history[interaction_partner, epoch] = i 
            else
                println("partner already had partner")
                h.interaction_partner_history[i, epoch] = 0
            end
        else 
            println("chose no")
            h.interaction_partner_history[i, epoch] = 0
        end
    end
    return 0
end


function interact!(h::Hive, epoch::UInt16, learning_rate::Float16)
    bee_ids = collect(1:h.n_bees)
    println("bee_ids is $(bee_ids)")
    for i in 1:h.n_bees
        if !(i in bee_ids)
            println("bee $(i) will be skipped")
            continue
        end
        interaction_partner = h.interaction_partner_history[i, epoch]
        println("interaction partner of bee $(i) is $(interaction_partner)")
        if interaction_partner == -1
            throw(ArgumentError("interaction partners not chosen for given epoch"))
        end
        if interaction_partner == 0
            println("bee does not interact")
            filter!(x -> x != i, bee_ids)
            h.interaction_results_history[i, epoch] = 2
            continue
        end
        accuracy_bee = h.accuracy_history[i, epoch]
        accuracy_partner = h.accuracy_history[interaction_partner, epoch]
        flattened_params_bee, restructor_bee = Flux.destructure(h.bee_list[i].brain)
        flattened_params_partner, restructor_partner = Flux.destructure(h.bee_list[interaction_partner].brain)
        #model_params_bee = Flux.params(h.bee_list[i].brain)
        #model_params_partner = Flux.params(h.bee_list[interaction_partner].brain)
        prefactor = learning_rate
        if accuracy_bee > accuracy_partner
            println("bee won")
            flattened_params_partner .-= prefactor .* (flattened_params_bee .- flattened_params_partner)
            h.bee_list[interaction_partner].brain = restructor_partner(flattened_params_partner)
            #model_params_partner = model_params_partner .- prefactor * (model_params_bee .- model_params_partner)
            h.interaction_results_history[i, epoch] = 1
            h.interaction_results_history[interaction_partner, epoch] = 0
        else
            flattened_params_bee .-= prefactor .* (flattened_params_partner .- flattened_params_bee)
            h.bee_list[i].brain = restructor_partner(flattened_params_bee)
            println("partner won")
            #model_params_bee = model_params_bee - prefactor * (model_params_partner - model_params_bee)
            h.interaction_results_history[i, epoch] = 0
            h.interaction_results_history[interaction_partner, epoch] = 1
        end
        filter!(x -> x != i, bee_ids)
        filter!(x -> x != interaction_partner, bee_ids)
    end
end

function save_data(raw_path::String, h::Hive, n_epochs=DEFAULTS[:N_EPOCHS])
    mkpath(raw_path)
    #epoch_ids = collect(1:h.epoch_index)
    epoch_ids = collect(1:n_epochs)
    accuracies_epoch_ids = collect(0:n_epochs)
    export_data(string(raw_path, "/loss_history", ".csv"), h.loss_history, h.n_bees, epoch_ids, "loss")
    export_data(string(raw_path, "/partner_history", ".csv"), h.loss_history, h.n_bees, epoch_ids, "partner")
    export_data(string(raw_path, "/result_history", ".csv"), h.loss_history, h.n_bees, epoch_ids, "result")
    export_data(string(raw_path, "/accuracy_history", ".csv"), h.accuracy_history, h.n_bees, accuracies_epoch_ids, "accuracy")
    return 0
end

function train_task!(h::Hive, data, n_epochs::UInt16 = DEFAULTS[:N_EPOCHS], n_labels = 10)
    learning_rate = DEFAULTS[:LEARNING_RATE]
    optimizer = Flux.Adam(learning_rate) 
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
    trainloader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((data[3], data[4]), batchsize=128)
    for bee in h.bee_list
        initial_accuracy = calc_accuracy(bee.brain, testloader, n_labels)
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
            accuracy = calc_accuracy(bee.brain, testloader, n_labels)
            h.accuracy_history[bee.id, (epoch + 1)] = accuracy
            bee.params_history[epoch] = deepcopy(Flux.params(bee.brain))
            println("Epoch = $epoch : Bee ID = $(bee.id) : Loss = $epoch_loss : Accuracy = $accuracy")
        end
    end
    save_data(RAW_PATH, h, n_epochs)
    return 0
end






"""
--------------------------------------------------------------------------
work in progress from here on
"""

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





"""
This function trains one individual bee.
At the moment the function is obsolete since the training already takes place in the run_simulation function
In case there will be some form of interaction between the bees during the training process this function will be used
"""
function train_bee!(bee::Bee, dataloader, loss_fn, optimizer, output_range)
    total_loss = 0.0
    for (x_batch, y_batch) in dataloader
        model = bee.brain
        grads = gradient(()->loss_fn(model(x_batch)[output_range[1], output_range[2]], y_batch), Flux.params(model))
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)
        total_loss += loss_fn(model(x_batch), y_batch)
    end
    return total_loss
end

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

function run()
    h = Hive(N_BEES, N_EPOCHS)
    data_full_mnist = prepare_MNIST(false, false) #the dataset is not normalized to 1
    #train_task!()
end