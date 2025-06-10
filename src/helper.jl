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


function export_data(file_path::String, data::Array, value_col_name::String)
    rows, cols = size(data)
    if ndims(data) != 2
        error("Expected data to be a 2D array (Matrix), but got a $(ndims(data))D array.")
    end
    epoch_ids = collect(1:cols)

    dt = DataFrame(data, :auto)
    rename!(dt, Symbol.(epoch_ids))
    dt[!, :id] = 1:rows
    dt_long = stack(dt, Not(:id))
    rename!(dt_long, Symbol.(["bee_id", "epoch_index", value_col_name]))
    CSV.write(file_path, dt_long)
    return 0
end

function vector_to_dataframe(file_path:: String, values::Vector{T}, column_name::Symbol) where T
    epoch_ids = collect(1:length(values))
    df = DataFrame(:epoch_id => epoch_ids, column_name => values)
    #df = DataFrame(epoch_index=1:length(values), column_name => values)
    CSV.write(file_path, df)
    return 0
end

function get_git_branch()
    return try
        read(`git rev-parse --abbrev-ref HEAD`, String) |> strip
    catch
        "unknown"
    end
end

function get_git_commit()
    return try
        read(`git rev-parse HEAD`, String) |> strip
    catch
        "unknown"
    end
end
