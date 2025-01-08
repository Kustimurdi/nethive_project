using Flux 

mutable struct Bee
    id::Integer
    brain::Flux.Chain
    #interaction_partner_history::Vector{Int}
    function Bee(id::Integer, brain::Flux.Chain)
        new(id,
        brain)
        #zeros(Int8, N_EPOCHS)
    end
end

mutable struct Hive
    n_bees::UInt16
    bee_list::Array{Bee}
    function Hive(n_bees::UInt16, brain::Flux.Chain)
        bee_list = Array{Bee}(undef, n_bees)
        for i = 1:n_bees
            bee_list[i] = Bee(UInt16(i), brain)
        end
        new(n_bees::UInt16,
        bee_list)
    end
end


"""
---------------------------------------------
Testing
---------------------------------------------
"""



hive1 = Hive(UInt16(5))
println(hive1.bee_list)
println(hive1.n_bees)