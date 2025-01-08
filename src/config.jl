using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--n_bees"
    help = "The number of neural networks in the hive"
    arg_type = Int
    default = 1
    nargs = 'A'
    "--n_epochs"
    help = "Number of epochs to train"
    arg_type = Int
    default = 100
    nargs = 'A'
end