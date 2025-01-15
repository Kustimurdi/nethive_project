include("arg_table.jl")
using .Defaults
using .ArgParseTable

# Parse arguments
s = create_arg_parse_settings()
parsed_args = parse_args(ARGS, s)


# Define constants
const N_BEES::UInt16 = haskey(parsed_args, "n_bees") ? UInt16(parsed_args["n_bees"]) : Config.DEFAULT_N_BEES
const N_EPOCHS::UInt16 = haskey(parsed_args, "n_epochs") ? UInt16(parsed_args["n_epochs"]) : Config.DEFAULT_N_EPOCHS
const INPUT_SIZE::Vector{UInt16} = haskey(parsed_args, "input_size") ? parsed_args["input_size"] : Config.DEFAULT_INPUT_SIZE
const OUTPUT_SIZE::UInt16 = haskey(parsed_args, "output_size") ? parsed_args["output_size"] : Config.DEFAULT_OUTPUT_SIZE
