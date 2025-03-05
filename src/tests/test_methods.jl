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
gillespie_train_task_with_epochs!(test_hive, trainloader=trainloader, testloader=testloader, n_epochs=5)