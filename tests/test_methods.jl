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


gillespie_testing(test_hive, trainloader=trainloader, testloader=testloader, n_epochs=5, lambda_Train=0, lambda_Interact=50)

##test influence of changing one weight
model = build_model_4()
x_train, y_train = prepare_mnist_dataset_1_channel(:train)
x_test, y_test = prepare_mnist_dataset_1_channel(:test)
trainloader = Flux.DataLoader((x_train, y_train), batchsize=128, shuffle=true)
testloader = Flux.DataLoader((x_test, y_test), batchsize=128, shuffle=true)
train_model!(model, trainloader)
weight1 = deepcopy(model[1].weight[1,1,1,1])
weight2 = deepcopy(model[1].weight[1,1,1,2])
weight3 = deepcopy(model[4].weight[1,1,1,1])
weight4 = deepcopy(model[20].weight[1,1])
weight5 = deepcopy(model[10].weight[1,1,1,1])



current_acc = calc_accuracy(model, testloader)
model[1].weight[1,1,1,1] = weight1
model[1].weight[1,1,1,2] = weight2
model[4].weight[1,1,1,1] = weight3
model[20].weight[1,1] = weight4
model[10].weight[1,1,1,1] = 3



sin_traindata = create_sin_dataset(5, 1, 10000)
sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
sin_testdata = create_sin_dataset(5, 1, 1000)
sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
sin_model = build_model_sin()

init_acc = calc_regression_accuracy(sin_model, sin_testloader, atol=DEFAULTS[:ACCURACY_ATOL])
for i in 1:30
    train_regression_model!(sin_model, sin_trainloader)
end
new_acc = calc_regression_accuracy(sin_model, sin_testloader, atol=0.1)


diff = sum(abs.(vec(sin_traindata[2]) .- vec(sin_model(sin_traindata[1]))))


scatter(vec(sin_traindata[1]), vec(sin_traindata[2]))
y_pred = sin_model(sin_traindata[1])
scatter(vec(sin_traindata[1]), vec(y_pred))

result = isapprox.(vec(sin_traindata[2]), vec(y_pred), atol=0.005)

num_true = count(result)
num_false = count(!, result)

num_true/10000

result_test = isapprox.(vec(sin_testdata[2]), vec(sin_model(sin_testdata[2])), atol=0.5)
num_test_true = count(result_test)

correct = 0
total = 0
for (xb, yb) in sin_testloader
    test_pred = sin_model(xb)
    batch_result = isapprox.(vec(yb), vec(test_pred), atol=0.005)
    num_true_batch = count(batch_result)
    correct += num_true_batch
    total += length(yb)
end

test_sin_data = create_sin_dataset(3, 2, 3000)
sinx = vec(test_sin_data[1])
siny = vec(test_sin_data[2])
scatter(sinx, siny)


##
sin_hive = Hive(UInt16(5), UInt16(5))
sin_traindata = create_sin_dataset(5, 1, 10000)
sin_trainloader = Flux.DataLoader((sin_traindata[1], sin_traindata[2]), batchsize=128, shuffle=true)
sin_testdata = create_sin_dataset(5, 1, 1000)
sin_testloader = Flux.DataLoader((sin_testdata[1], sin_testdata[2]), batchsize=128, shuffle=true)
gillespie_regression!(sin_hive, n_epochs=UInt16(5), trainloader=sin_trainloader, testloader=sin_testloader)


run_regression(n_bees=UInt16(5), n_epochs=UInt16(40), n_peaks=5, which_peak=1, trainsetsize=10000, testsetsize=1000)

sinmodel = build_model_sin()
train_regression_model!(sinmodel)
punish_regression_model!(sinmodel, sin_trainloader, 0.005)

run_testing_first_bee(n_bees=UInt16(5), n_epochs=UInt16(40), n_peaks= 5, which_peak=1, trainsetsize=10000, testsetsize=1000)
