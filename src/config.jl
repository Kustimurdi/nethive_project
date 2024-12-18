using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--n_networks"
    help = "The number of neural networks in the hive"
    arg_type = Int
    default = 1
    nargs = 'A'
    "--n_neurons_per_layer"
    help = ""
    arg_type = Int
    default = 21
    nargs = 'A'
    "--n_episodes"
    help = "Number of episodes to train"
    arg_type = Int
    default = 100
    nargs = 'A'
end