using ArgParse
using Flux
using MLDatasets
using CSV
using DataFrames
using Dates


function export_data(file_path::String, data::Array, n_ids::Integer, epoch_ids::Vector{Int64}, value_col_name::String)
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
    CSV.write(string(raw_path, "/parameters.csv"), dt_long, writeheader=true)
    return 0
end
