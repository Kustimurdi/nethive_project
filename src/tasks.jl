abstract type Task end

struct RegressionTask <: Task end
struct LinearRegressionTask <: Task end

struct ClassificationTask <: Task
    input_size::AbstractVector{<:Integer}
    output_size::UInt16
end

struct NoTask <: Task end  # Placeholder for bees without tasks

function train_model!(model, dataloader, task::Task; learning_rate=DEFAULTS[:LEARNING_RATE])
    optimizer = Flux.Adam(learning_rate)
    total_batch_loss = 0.0
    n_batches = 0

    loss_fn(x, y) = compute_task_loss(model, x, y, task)  # Unified loss function

    for (x_batch, y_batch) in dataloader
        Flux.train!(loss_fn, Flux.params(model), [(x_batch, y_batch)], optimizer)

        total_batch_loss += loss_fn(x_batch, y_batch)
        n_batches += 1
    end

    return total_batch_loss / max(n_batches, 1)
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

function calc_accuracy(model, dataloader, task::Task; acc_sigma=1.0, num_batches::Int=typemax(Int))
    if task isa RegressionTask || task isa LinearRegressionTask
        return calc_gaussian_regression_accuracy(model, dataloader, sigma=acc_sigma, num_batches=num_batches)
    
    elseif task isa ClassificationTask
        return calc_accuracy_labels(model, dataloader, n_labels=task.output_size, num_batches=num_batches)
    
    else
        error("Unknown task type: $task")
    end
end


function calc_loss(model, dataloader, task::Task)
    total_batch_loss = 0.0
    n_batches = 0

    for (x_batch, y_batch) in dataloader
        loss = compute_task_loss(model, x_batch, y_batch, task)
        total_batch_loss += loss
        n_batches += 1
    end

    return total_batch_loss / max(n_batches, 1)  # Avoid division by zero
end

function compute_task_loss(model, x, y, ::RegressionTask)
    return Flux.Losses.mse(model(x), y)
end

function compute_task_loss(model, x, y, ::LinearRegressionTask)
    return Flux.Losses.mse(model(x), y)
end

function compute_task_loss(model, x, y, ::ClassificationTask)
    return Flux.Losses.crossentropy(model(x), y)  # One-hot encoded classification
end

function compute_task_loss(model::Nothing=nothing, x::Nothing=nothing, y::Nothing=nothing, ::NoTask)
    return 0.0  # No task, no meaningful loss
end


"""
------------------------------------------------------------
old training function
------------------------------------------------------------
"""

function train_regression_model!(model, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
    loss_fn(x, y) = Flux.Losses.mse(model(x), y)
    optimizer = Flux.Adam(learning_rate)
    total_batch_loss = 0.0
    n_batches = 0
    for (x_batch, y_batch) in dataloader
        Flux.train!(loss_fn, Flux.params(model), [(x_batch, y_batch)], optimizer)

        total_batch_loss += loss_fn(x_batch, y_batch)
        n_batches += 1
    end

    return total_batch_loss / n_batches  # Return average loss
end

"""
the function @calc_accuracy calculates the accuracy of a neural network given by @model by averaging the results of the model on the dataset given in @dataloader
"""
function calc_regression_accuracy(model, dataloader; atol=0.005, num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = model(x_batch)
        truths = y_batch       # Get true labels
        result = isapprox.(vec(preds), vec(truths), atol=atol)

        num_true = count(result)
        num_false = count(!, result)

        correct += num_true
        total += length(truths)
    end
    return correct / total
end

#new regression accuracy for smoother transitions: look at difference between output and truth and 
#calculate output of some score function eg gaussian dist

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

function calc_regression_loss(model, dataloader)
    loss_fn(x, y) = Flux.Losses.mse(model(x), y)
    total_batch_loss = 0.0
    n_batches = 0
    for (x_batch, y_batch) in dataloader
        total_batch_loss += loss_fn(x_batch, y_batch)
        n_batches += 1
    end

    return total_batch_loss / n_batches  # Return average loss
end

function calc_accuracy(bee::Bee, dataloader; num_batches::Int=typemax(Int))
    task_type = bee.task_type
    
    # For Regression tasks
    if task_type isa RegressionTask || task_type isa LinearRegressionTask
        return calc_gaussian_regression_accuracy(bee.brain, dataloader, num_batches=num_batches)
    
    # For Classification tasks
    elseif task_type isa ClassificationTask
        return calc_accuracy_labels(bee.brain, dataloader, n_labels=task_type.output_size, num_batches=num_batches)
    
    else
        error("Unknown task type: $task_type")
    end
end

function train_model!(bee::Bee, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
    # If bee has no task, skip training
    if bee.task_type isa NoTask
        println("Bee $bee.id has no task and will not be trained.")
        return 0.0  # No training performed, return 0 loss
    end
    
    # Loss function depending on task type
    if bee.brain isa Flux.Chain  # Check if bee has a valid brain model (neural network)
        task_type = bee.task_type

        if task_type isa RegressionTask
            # Regression task (e.g., using MSE loss)
            loss_fn(x, y) = Flux.Losses.mse(bee.brain(x), y)

        elseif task_type isa LinearRegressionTask
            # Linear regression task (using MSE loss but adjusted model if needed)
            loss_fn(x, y) = Flux.Losses.mse(bee.brain(x), y)

        elseif task_type isa ClassificationTask
            # Classification task (e.g., using cross-entropy loss)
            loss_fn(x, y) = Flux.Losses.logitcrossentropy(bee.brain(x), y)

        else
            error("Unknown task type: $task_type")
        end

        optimizer = Flux.Adam(learning_rate)
        total_batch_loss = 0.0
        n_batches = 0

        # Training loop
        for (x_batch, y_batch) in dataloader
            Flux.train!(loss_fn, Flux.params(bee.brain), [(x_batch, y_batch)], optimizer)

            total_batch_loss += loss_fn(x_batch, y_batch)
            n_batches += 1
        end

        return total_batch_loss / n_batches  # Return average loss
    else
        error("Bee brain is not properly initialized.")
    end
end

