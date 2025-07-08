function TaskConfig(parsed_args)
    task_type = parsed_args["task_type"]
    #task_type = Symbol(parsed_args["task_type"])

    if task_type == :regression || task_type == :linear_regression
        return RegressionTaskConfig(
            parsed_args["regression_n_peaks"],
            parsed_args["regression_which_peak"],
            parsed_args["trainset_size"],
            parsed_args["testset_size"]
        )
    elseif task_type == :classification
        #return ClassificationTaskConfig(
        #    parsed_args["num_classes"],
        #    parsed_args["class_distribution"]
        #)
        throw(ArgumentError("Classification task configuration not implemented"))
    elseif task_type == :custom_classification
        return CustomClassificationTaskConfig(
            parsed_args["features_dimension"],
            parsed_args["n_classes"],
            parsed_args["n_per_class_train"],
            parsed_args["n_per_class_test"],
            parsed_args["class_center_radius"],
            parsed_args["sampling_gauss_sigma"]
        )
    elseif task_type == :none || task_type == :NoTask
        return NoTaskConfig()
    else
        throw(ArgumentError("Unsupported task type: $task_type"))
    end
end

function get_task_instance(symbol::Symbol)::AbstractTask
    if symbol == :regression
        return RegressionTask()
    elseif symbol == :linear_regression
        return LinearRegressionTask()
    elseif symbol == :custom_classification
        return CustomClassificationTask()
    elseif symbol == :none
        return NoTask()
    else
        error("Unsupported task symbol: $symbol")
    end
end

function get_qgm_instance(symbol::Symbol)::QueenGeneMethod
    if symbol == :accuracy
        return QueenGeneFromAccuracy()
    elseif symbol == :loss
        return QueenGeneFromLoss()
    elseif symbol == :incremental
        qgir = DEFAULTS[:queen_gene_incremental_rate]
        qgdr = DEFAULTS[:queen_gene_decremental_rate]
        return QueenGeneIncremental(qgir, qgdr)
    else
        error("Unsupported queen gene method symbol: $symbol")
    end
end

function create_dataset(task::RegressionTask, config::RegressionTaskConfig)
    train_data = create_sin_dataset(config.n_peaks, config.which_peak, config.trainset_size)
    test_data = create_sin_dataset(config.n_peaks, config.which_peak, config.testset_size)
    return Flux.DataLoader((train_data[1], train_data[2]), batchsize=128, shuffle=true),
            Flux.DataLoader((test_data[1], test_data[2]), batchsize=128, shuffle=true),
            train_data,
            test_data
end

function create_dataset(task::LinearRegressionTask, config::RegressionTaskConfig)
    train_data = create_linear_dataset(config.trainset_size)
    test_data = create_linear_dataset(config.testset_size)
    return Flux.DataLoader((train_data[1], train_data[2]), batchsize=128, shuffle=true),
            Flux.DataLoader((test_data[1], test_data[2]), batchsize=128, shuffle=true),
            train_data,
            test_data
end

function create_dataset(task::CustomClassificationTask, config::CustomClassificationTaskConfig; batchsize=128)
    centers = generate_orthogonal_centers(config.n_classes, config.features_dimension, config.class_center_radius)
    train_features, train_labels = generate_gaussian_classification_data(centers, config.sampling_gauss_sigma, config.n_per_class_train)
    test_features, test_labels = generate_gaussian_classification_data(centers, config.sampling_gauss_sigma, config.n_per_class_test)
    train_labels = Flux.onehotbatch(train_labels, 1:config.n_classes)
    test_labels = Flux.onehotbatch(test_labels, 1:config.n_classes)
    return Flux.DataLoader((train_features', train_labels), batchsize=batchsize, shuffle=true),
            Flux.DataLoader((test_features', test_labels), batchsize=batchsize, shuffle=true),
            (train_features', train_labels),
            (test_features', test_labels)
end

function create_dataset(task::ClassificationTask, config::ClassificationTaskConfig)
    throw(ArgumentError("Classification dataset creation not implemented"))
end

function create_dataset(task::NoTask, config::NoTaskConfig)
    return nothing, nothing, nothing, nothing
end

function create_linear_dataset(setsize::Int)
    # Simple linear regression data (y = mx + b)
    x = rand(setsize) * 10
    m, b = 2.0, 5.0  # slope and intercept
    y = m * x .+ b + randn(setsize) * 0.5  # Adding noise
    return reshape(Float32.(x), 1, :), reshape(Float32.(y), 1, :)
end

function create_sin_dataset(n_peaks, which_peak, setsize::Int)
    features = rand(setsize) * pi *n_peaks |> x -> reshape(x, 1, :)
    temp = deepcopy(features)
    temp[(temp .< (which_peak - 1)*pi) .| (temp .> which_peak*pi)] .= 0
    labels = abs.(sin.(temp)) * 10
    return Float32.(features), Float32.(labels)
end

function sample_gaussian_around(x::Vector, sigma::Float64, n::Int)
    d = length(x)
    cov = (sigma^2) * I
    dist = MvNormal(x, cov)
    return rand(dist, n)
end


"""
    generate_gaussian_classification_data(centers::Vector{Vector{Float64}}, sigma::Float64, n_per_class::Int)

Generates synthetic classification data.

- `centers`: a vector of center vectors, one per class
- `sigma`: standard deviation for Gaussian sampling
- `n_per_class`: number of samples per class

Returns:
- `X`: matrix of features (n_total × d)
- `y`: vector of labels (Int) (n_total)
"""
function generate_gaussian_classification_data(centers::Vector{Vector{Float64}}, sigma::Float64, n_per_class::Int)
    d = length(centers[1])
    k = length(centers)  # number of classes

    X = Matrix{Float64}(undef, 0, d)
    y = Int[]

    for (label, center) in enumerate(centers)
        cov = (sigma^2) * I
        dist = MvNormal(center, cov)
        samples = rand(dist, n_per_class)'  # n_per_class × d
        X = vcat(X, samples)
        append!(y, fill(label, n_per_class))
    end

    return X, y
end

"""
    generate_orthogonal_centers(n_classes::Int, d_features::Int; radius=5.0)

Returns a vector of class centers placed on orthogonal axes.
"""
function generate_orthogonal_centers(n_classes::Int, d_features::Int, radius)
    @assert n_classes <= d_features "You need at least as many dimensions as classes"

    return [radius * unit_vector(i, d_features) for i in 1:n_classes]
end

# Helper: returns a unit vector with 1.0 at index `i`
unit_vector(i, dim) = [j == i ? 1.0 : 0.0 for j in 1:dim]
