abstract type AbstractTask end

struct RegressionTask <: AbstractTask end
struct LinearRegressionTask <: AbstractTask end

struct ClassificationTask <: AbstractTask
    input_size::AbstractVector{<:Integer}
    output_size::UInt16
end

struct NoTask <: AbstractTask end  # Placeholder for bees without tasks



abstract type AbstractTaskConfig end

mutable struct NoTaskConfig <: AbstractTaskConfig end
mutable struct RegressionTaskConfig <: AbstractTaskConfig
    n_peaks::Int
    which_peak::Int
    trainset_size::Int
    testset_size::Int
end

mutable struct ClassificationTaskConfig <: AbstractTaskConfig
    num_classes::Int
    class_distribution::Vector{Float64}
end
