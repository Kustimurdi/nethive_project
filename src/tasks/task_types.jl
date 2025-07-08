abstract type AbstractTask end

struct RegressionTask <: AbstractTask end
struct LinearRegressionTask <: AbstractTask end

struct ClassificationTask <: AbstractTask
    input_size::AbstractVector{<:Integer}
    output_size::UInt16
end

struct CustomClassificationTask <: AbstractTask end

struct NoTask <: AbstractTask end  # Placeholder for bees without tasks



abstract type AbstractTaskConfig end

mutable struct NoTaskConfig <: AbstractTaskConfig end
mutable struct LinearRegressionTaskConfig <: AbstractTaskConfig end
mutable struct RegressionTaskConfig <: AbstractTaskConfig
    n_peaks::Int
    which_peak::Int
    trainset_size::Int
    testset_size::Int
end

mutable struct ClassificationTaskConfig <: AbstractTaskConfig
    num_classes::Int
    class_distribution::Vector{Float64}
    input_size::AbstractVector{<:Integer}
    output_size::UInt16
end

mutable struct CustomClassificationTaskConfig <: AbstractTaskConfig
    features_dimension::Int
    n_classes::Int
    n_per_class_train::Int
    n_per_class_test::Int
    class_center_radius::Float64
    sampling_gauss_sigma::Float64
end
