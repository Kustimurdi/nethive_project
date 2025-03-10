using Revise
using Plots
using LinearAlgebra
using Random
using ArgParse
using Flux
using MLDatasets
using CSV
using DataFrames
using Dates
using Distributions
using Images
using Test
using Serialization
using Logging
using LoggingExtras
global_logger(ConsoleLogger(stderr, Logging.Info))
#global_logger(ConsoleLogger(stdout))


function export_data(file_path::String, data::Array, n_ids::Integer, epoch_ids::Vector{Int64}, value_col_name::String)
    if ndims(data) != 2
        error("Expected data to be a 2D array (Matrix), but got a $(ndims(data))D array.")
    end
    if size(data, 1) != n_ids
        error("Mismatch: `n_ids` ($n_ids) does not match data row count ($(size(data,1))).")
    end
    if length(epoch_ids) != size(data, 2)
        error("Mismatch: `epoch_ids` has $(length(epoch_ids)) elements, but data has $(size(data,2)) columns.")
    end
    dt = DataFrame(data, :auto)
    rename!(dt, Symbol.(epoch_ids))
    dt[!, :id] = 1:n_ids
    dt_long = stack(dt, Not(:id))
    rename!(dt_long, Symbol.(["bee_id", "epoch_index", value_col_name]))
    CSV.write(file_path, dt_long)
    return 0
end


"""
Save the parameters of the simulations
"""
function save_params(parsed_args, raw_path::String)
    mkpath(raw_path)
    print("Data path: ", raw_path, "\n")

    dt = DataFrame(parsed_args)
    println(nrow(dt))
    println(dt)

    dt[!, :id] = 1:nrow(dt)
    insertcols!(dt, 1, :dataset_name => DATASET_NAME)

    dt_long = stack(dt, Not(:id))
    select!(dt_long, Not(:id))
    rename!(dt_long, Symbol.(["id", "value"]))

    git_branch_row = DataFrame(id="git branch", value=GIT_BRANCH)
    append!(dt_long, git_branch_row)

    git_commit_id_row = DataFrame(id="git commit id", value=GIT_COMMIT)
    append!(dt_long, git_commit_id_row)

    CSV.write(string(raw_path, "/parameters.csv"), dt_long, writeheader=true)
    return 0
end
