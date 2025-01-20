using Test

#include("../methods.jl")
#include("../definitions.jl")

using .Methods
using .Definitions

#@testset "Test run_simulation" begin
    h1 = Definitions.Hive()
    learning_rate = Float16(0.01)
    optimizer = Flux.Adam(learning_rate) 
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y) 
    data = prepare_MNIST()
    trainloader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
    testloader = Flux.DataLoader((data[3], data[4]), batchsize=128)
    bee1 = h1.bee_list[1]
    @show typeof(bee1)
    #@show Methods.Definitions.Bee
    one_epoch_loss = Methods.train_bee!(bee1, trainloader, loss_fn, optimizer)
    #h1.bee_list[1].loss_history[1] = one_epoch_loss
    #export_bee_data("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/testrun1", h1.bee_list[1])
    #export_hive_data("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/testrun1", h1)
#end