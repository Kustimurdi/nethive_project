module ArgParseTable 

include("defaults.jl") 
using .Defaults
using ArgParse

function create_arg_parse_settings()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--n_bees"
        help = "The number of neural networks in the hive"
        arg_type = UInt16
        default = Defaults.DEFAULT_N_BEES 
        nargs = 'A'
        "--n_epochs"
        help = "Number of epochs to train"
        arg_type = UInt16
        default = Defaults.DEFAULT_N_EPOCHS
        nargs = 'A'
        "--input_size"
        help = "Size of the input layer of the neural networks of the @Bee objects"
        arg_type = Array{UInt16}
        default = Defaults.DEFAULT_INPUT_SIZE
        nargs = 'A'
        "--output_size"
        help = "Size of the ouput layer of the neural networks of the @Bee objects"
        arg_type = UInt16
        default = Defaults.DEFAULT_OUTPUT_SIZE
        nargs = 'A'
    end
    
    return s

end 

export create_arg_parse_settings

end