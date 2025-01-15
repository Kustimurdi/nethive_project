

"""
------------------------------------------
Testing
------------------------------------------
"""



model = build_brain([UInt16(28), UInt16(28), UInt16(1)], UInt16(10))
println("hello")
#h = Hive(UInt16(3), UInt16(10), build_brain(), [UInt16(28), UInt16(28), UInt16(1)], UInt16(10))
h = Hive(UInt16(3), UInt16(10))

run_simulation(h, UInt8(3))

"""
xs = rand(28, 28, 1, 10)
model(xs)

example = [1, [2,2,2]]
size(example)

data = prepare_MNIST()

size(data[1])
size(xs)
trainloader = Flux.DataLoader((data[1], data[2]), batchsize=128, shuffle=true)
trainloader
i = 0
for (xbatch, ybatch) in trainloader

    println(i)
    i += 1
end
"""