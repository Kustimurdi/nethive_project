
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

function interact!(sub_bee::Bee, dom_bee::Bee, interaction_prefactor)
    prefactor = interaction_prefactor
    flattened_params_sub_bee, restructor_sub_bee = Flux.destructure(sub_bee.brain)
    flattened_params_dom_bee, restructor_dom_bee = Flux.destructure(dom_bee.brain)
    flattened_params_sub_bee .-= prefactor .* (flattened_params_dom_bee .- flattened_params_sub_bee)
    sub_bee.brain = restructor_sub_bee(flattened_params_sub_bee)
    return 0
end

function train_model_manually!(model, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y)
    optimizer = Flux.Adam(learning_rate)

    total_loss = 0.0
    num_batches = 0

    for (x_batch, y_batch) in dataloader
        # Compute loss inside gradient calculation
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x_batch)
            loss_fn(y_hat, y_batch)
        end
        
        # Apply gradients
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)

        # Track loss
        total_loss += loss
        num_batches += 1
    end

    return total_loss / num_batches  # Return average loss
end