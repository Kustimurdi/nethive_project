abstract type QueenGeneMethod end

struct QueenGeneFromAccuracy <: QueenGeneMethod end
struct QueenGeneFromLoss <: QueenGeneMethod end
struct QueenGeneIncremental <: QueenGeneMethod
    increment_value::Float64
end

function compute_queen_gene(bee::Bee, dataloader, task::Task, method::QueenGeneFromAccuracy)
    return calc_accuracy(bee.brain, dataloader, task)
end

function compute_queen_gene(bee::Bee, dataloader, task::Task, method::QueenGeneFromLoss)
    loss = calc_loss(bee.brain, dataloader, task)
    return (1/(loss+1))  # Use the loss function
end

function compute_queen_gene(bee::Bee, method::QueenGeneIncremental, dataloader::Nothing=nothing, task::Nothing=nothing)
    return bee.queen_gene + method.increment_value  # Increment by fixed value
end
