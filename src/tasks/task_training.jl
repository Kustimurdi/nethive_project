function train_model!(model, dataloader, task::AbstractTask; learning_rate)
    if task isa NoTask
        return 0.0  # No task, no training
    end
    #optimizer = Flux.Adam(learning_rate)
    optimizer = Flux.Descent(learning_rate)
    total_batch_loss = 0.0
    n_batches = 0

    loss_fn(x, y) = compute_task_loss(task, model, x, y)  # Unified loss function

    for (x_batch, y_batch) in dataloader
        Flux.train!(loss_fn, Flux.params(model), [(x_batch, y_batch)], optimizer)

        total_batch_loss += loss_fn(x_batch, y_batch)
        n_batches += 1
    end

    return total_batch_loss / max(n_batches, 1)
end

function punish_model!(model::Flux.Chain, dataloader, task::AbstractTask; punish_rate)
    if task isa NoTask
        return 0.0  # No task, no punishment
    end
    loss_fn(x, y) = compute_task_loss(task, model, x, y)
    total_batch_loss = 0.0
    n_batches = 0
    for (x_batch, y_batch) in dataloader
        grads = Flux.gradient(() -> loss_fn(x_batch, y_batch), Flux.params(model))
        for p in Flux.params(model)
            p .= p .+ punish_rate .* grads[p]
        end

        total_batch_loss += loss_fn(x_batch, y_batch)
        n_batches +=1
    end
    return total_batch_loss/n_batches
end

function calc_regression_accuracy(model, dataloader; atol=0.005, num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = model(x_batch)
        truths = y_batch
        result = isapprox.(vec(preds), vec(truths), atol=atol)

        num_true = count(result)
        num_false = count(!, result)

        correct += num_true
        total += length(truths)
    end
    return correct / total
end

function calc_gaussian_regression_accuracy(model, dataloader; sigma=1.0, num_batches::Int=typemax(Int))
    total_accuracy = 0.0
    total_samples = 0
    
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = model(x_batch)

        diffs = vec(preds) .- vec(y_batch)  # Calculate the differences
        
        # Apply the Gaussian function to each difference
        accuracies = exp.(.-(diffs .^ 2) / (2 * sigma^2))
        
        total_accuracy += sum(accuracies)
        total_samples += length(accuracies)
    end
    
    return total_accuracy / total_samples  # Return the average accuracy
end

function calc_accuracy_labels(model, dataloader; n_labels=10, num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = Flux.onecold(model(x_batch), 0:(n_labels - 1))  # Get predicted labels
        truths = Flux.onecold(y_batch, 0:n_labels - 1)          # Get true labels
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end

function calc_accuracy_one_cold(model, dataloader; num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = Flux.onecold(model(x_batch))
        truths = Flux.onecold(y_batch)
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end


function calc_accuracy(model, dataloader, task::AbstractTask; acc_sigma=1.0, num_batches::Int=typemax(Int))
    if task isa RegressionTask || task isa LinearRegressionTask
        return calc_gaussian_regression_accuracy(model, dataloader, sigma=acc_sigma, num_batches=num_batches)
    
    elseif task isa ClassificationTask
        return calc_accuracy_labels(model, dataloader, n_labels=task.output_size, num_batches=num_batches)
    
    elseif task isa CustomClassificationTask
        return calc_accuracy_one_cold(model, dataloader, num_batches=num_batches)

    elseif task isa NoTask
        return 0.0  # No task, no meaningful accuracy

    else
        error("Unknown task type: $task")
    end
end


function calc_loss(model, dataloader, task::AbstractTask)
    total_batch_loss = 0.0
    n_batches = 0

    for (x_batch, y_batch) in dataloader
        loss = compute_task_loss(task, model, x_batch, y_batch)
        total_batch_loss += loss
        n_batches += 1
    end

    return total_batch_loss / max(n_batches, 1)  # Avoid division by zero
end

function compute_task_loss(task::RegressionTask, model::Flux.Chain, x, y)
    return Flux.Losses.mse(model(x), y)
end

function compute_task_loss(task::LinearRegressionTask, model, x, y)
    return Flux.Losses.mse(model(x), y)
end

function compute_task_loss(::ClassificationTask, model, x, y)
    return Flux.Losses.crossentropy(model(x), y)  # One-hot encoded classification
end

function compute_task_loss(::CustomClassificationTask, model, x, y)
    return Flux.Losses.logitcrossentropy(model(x), y) # Already applies softmax
end

function compute_task_loss(task::NoTask, model::Nothing=nothing, x::Nothing=nothing, y::Nothing=nothing)
    return 0.0  # No task, no meaningful loss
end


function punish_model_resetting!(bee::Bee, task_config::AbstractTaskConfig)
    bee.brain = build_model(task_config)
    return nothing
end