
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

function prepare_cifar10_dataset_3_channels(split::Symbol; subset=nothing)
    dataset = MLDatasets.CIFAR10(split)
    images = Float32.(dataset.features) / 255.0
    labels = dataset.targets .+ 1

    mean_img = mean(images, dims=(1,2,3))
    std_img = std(images, dims=(1,2,3))
    images = (images .- mean_img) ./ std_img

    onehot_labels = Flux.onehotbatch(labels, 1:10)

    if subset !== nothing
        images = images[:, :, :, 1:subset]
        onehot_labels = onehot_labels[:, 1:subset]
    end
    
    return images, onehot_labels
end

function prepare_svhn2_dataset_3_channels(split::Symbol; subset=nothing)
    dataset = MLDatasets.SVHN2(split)
    images = Float32.(dataset.features) / 255.0
    labels = dataset.targets

    onehot_labels = Flux.onehotbatch(labels, 1:10)

    if subset !== nothing
        images = images[:, :, :, 1:subset]
        onehot_labels = onehot_labels[:, 1:subset]
    end
    
    return images, onehot_labels
end

function prepare_mnist_dataset_3_channels(split::Symbol; subset=nothing)
    dataset = MLDatasets.MNIST(split)
    processed_images, onehot_labels = prepare_greyscale_28x28_dataset(dataset; subset=subset)
    return processed_images, onehot_labels
end

function prepare_fashionmnist_dataset_3_channels(split::Symbol; subset=nothing)
    dataset = MLDatasets.FashionMNIST(split)
    processed_images, onehot_labels = prepare_greyscale_28x28_dataset(dataset; subset=subset)
    return processed_images, onehot_labels
end


function prepare_greyscale_28x28_dataset(dataset; subset=nothing)
    images = dataset.features
    labels = dataset.targets .+ 1  # Ensure 1-based indexing

    num_samples = subset !== nothing ? subset : size(images, 3)
    
    # Preallocate an array instead of concatenating
    processed_images = Array{Float32, 4}(undef, 32, 32, 3, num_samples)

    # Process each image and store it in the preallocated array
    for i in 1:num_samples
        #img = Float32.(images[:,:,i])
        img = Float32.(images[:, :, i]) / 255.0  # Normalize
        img_padded = parent(padarray(img, Fill(0, (2, 2))))  # Zero-pad to 32x32
        processed_images[:, :, 1, i] .= img_padded  # Copy image into 1st channel
        processed_images[:, :, 2, i] .= img_padded  # Copy into 2nd channel
        processed_images[:, :, 3, i] .= img_padded  # Copy into 3rd channel
    end

    # Convert labels to one-hot encoding
    onehot_labels = Flux.onehotbatch(labels[1:num_samples], 1:10)

    return processed_images, onehot_labels
end

function load_nn_state(raw_net_path::String, epoch_index::Int)
    brains = deserialize(raw_net_path)
    return brains
end