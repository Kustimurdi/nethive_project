# Parse arguments
function parse_args(args)
    parsed_args = ArgParse.parse_args(args, arg_table)

    # Convert string values to Symbol where needed
    parsed_args["task_type"] = Symbol(parsed_args["task_type"])
    parsed_args["queen_gene_method"] = Symbol(parsed_args["queen_gene_method"])

    return parsed_args
end


