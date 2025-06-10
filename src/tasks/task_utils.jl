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