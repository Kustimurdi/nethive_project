abstract type QueenGeneMethod end

struct QueenGeneFromAccuracy <: QueenGeneMethod end
struct QueenGeneFromLoss <: QueenGeneMethod end
struct QueenGeneIncremental <: QueenGeneMethod
    increment_value::Float64
    decrement_value::Float64
end

function compute_queen_gene(bee::Bee, method::QueenGeneFromAccuracy, dataloader, task::AbstractTask, acc_sigma)
    return calc_accuracy(bee.brain, dataloader, task, acc_sigma=acc_sigma)
end

function compute_queen_gene(bee::Bee, method::QueenGeneFromLoss, dataloader, task::AbstractTask, acc_sigma)
    loss = calc_loss(bee.brain, dataloader, task)
    return (1/(loss+1))  # Use the loss function
end

function compute_queen_gene(bee::Bee, method::QueenGeneIncremental, dataloader=nothing, task=nothing, acc_sigma=nothing)
    return bee.queen_gene 
    #return bee.queen_gene + method.increment_value  # Increment by fixed value
end

function compute_queen_gene(bee::Bee, method::QueenGeneMethod, dataloader, task, acc_sigma)
    throw(ArgumentError("QueenGeneMethod $(typeof(method)) not implemented."))
end
