# Parse arguments
function cparse_args(arg_table::ArgParseSettings; args::Vector{String} = ARGS)
    parsed_args = ArgParse.parse_args(args, arg_table)

    # Convert string values to Symbol where needed
    parsed_args["task_type"] = Symbol(parsed_args["task_type"])
    parsed_args["queen_gene_method"] = Symbol(parsed_args["queen_gene_method"])

    return parsed_args
end

function parse_args_with_args_file(s::ArgParseSettings)
    temp_args = parse_args(s)
    if haskey(temp_args, "args_file") && temp_args["args_file"] != ""
        arg_lines = readlines(temp_args["args_file"])
        push!(Base.ARGS, arg_lines...)
        return parse_args(s)
    end
    return temp_args
end

