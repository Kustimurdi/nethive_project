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

function create_sin_dataset(n_peaks, which_peak, setsize::Int)
    features = rand(setsize) * pi *n_peaks |> x -> reshape(x, 1, :)
    temp = deepcopy(features)
    temp[(temp .< (which_peak - 1)*pi) .| (temp .> which_peak*pi)] .= 0
    labels = abs.(sin.(temp)) * 10
    return features, labels
end