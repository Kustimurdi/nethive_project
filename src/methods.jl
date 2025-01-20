module Methods

using Flux, MLDatasets, BSON
using .Definitions

export prepare_MNIST, calc_accuracy, run_simulation, export_bee_data, export_hive_data, train_bee!

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

function calc_accuracy(model, dataloader, num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        preds = onecold(model(x_batch), 0:9)  # Get predicted labels
        truths = onecold(y_batch, 0:9)       # Get true labels
        correct += sum(preds .== truths)
        total += length(truths)
    end
    return correct / total
end
 
function train_bee!(bee::Bee, dataloader, loss_fn, optimizer)
    total_loss = 0.0
    for (x_batch, y_batch) in dataloader
        model = bee.brain
        grads = gradient(()->loss_fn(model(x_batch), y_batch), Flux.params(model))
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)
        total_loss += loss_fn(model(x_batch), y_batch)
    end
    return total_loss
end

function run_simulation(h::Hive, n_epochs::UInt16)
    println("bis hierhin")
    learning_rate = Float16(0.01)
    optimizer = Flux.Adam(learning_rate) 
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
    data = prepare_MNIST()
    trainloader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((data[3], data[4]), batchsize=128)
    
    for epoch = 1:n_epochs
        for bee in h.bee_list
            epoch_loss = 0.0
            for (x_batch, y_batch) in trainloader
                model = bee.brain
                grads = gradient(()->loss_fn(model(x_batch), y_batch), Flux.params(model))
                Flux.Optimise.update!(optimizer, Flux.params(model), grads)
                epoch_loss += loss_fn(model(x_batch), y_batch)
            end
            accuracy = calc_accuracy(bee.brain, testloader)
            bee.accuracy_history[epoch] = accuracy
            bee.loss_history[epoch] = epoch_loss
            bee.params_history[epoch] = deepcopy(Flux.params(bee.brain))
            println("Epoch = $epoch : Bee ID = $(bee.id) : Loss = $epoch_loss : Accuracy = $accuracy")
        end
    end
    return 0
end

function export_bee_data(file_path::String, bee::Bee)
    bee_data_file = joinpath(file_path, "bee_$(bee.id)_data.bson")
    BSON.@save bee_data_file 
        params_history=bee.params_history 
        loss_history=bee.loss_history 
        accuracy_history=bee.accuracy_history
end

function export_hive_data(file_path::String, hive::Hive)
    for bee in hive.bee_list
        export_bee_data(file_path, bee)
    end
end

end

