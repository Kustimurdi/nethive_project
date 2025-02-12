## testing trainloader
m = Flux.Chain(Dense(2 => 3, tanh),
softmax
)
data = prepare_MNIST() 
datax = rand(2, 100)
datay = rand(0:9, 100)
trainloader = Flux.DataLoader((datax, datay), batchsize=32, shuffle=true)
#result = calc_accuracy(m, trainloader, 3)
println(result)
##

## testing run_simulation and the interaction
h1 = Hive()
learning_rate = Float16(0.01)
optimizer = Flux.Adam(learning_rate) 
loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
data = prepare_MNIST()
trainloader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
testloader = Flux.DataLoader((data[3], data[4]), batchsize=128)
run_simulation(h1, UInt16(3), prepare_CIFAR10(), 10)
#save_data(RAW_PATH, h1)
##

## testing choose_partner
println(h1.interaction_partner_history)
h2 = Hive(UInt16(6))
h2.accuracy_history[:, :] .= 0.5
choose_partner!(h2, UInt16(3))
##

## testing CIFAR10
optimizer = Flux.Adam(Float16(0.01)) 
loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
cifar_bee = Bee(1)
cifardata = prepare_CIFAR10()
cifarloader = Flux.DataLoader((cifardata[1], cifardata[2]), batchsize=128, shuffle=true)
testloader = Flux.DataLoader((cifardata[3], cifardata[4]), batchsize=128)
for i in 1:DEFAULTS[:N_EPOCHS]
    train_bee!(cifar_bee, cifarloader, loss_fn, optimizer)

    println("Epoch = $epoch : Bee ID = $(bee.id) : Loss = $epoch_loss : Accuracy = $accuracy")
end
##


##
h3 = Hive(UInt16(10), UInt16(10))

