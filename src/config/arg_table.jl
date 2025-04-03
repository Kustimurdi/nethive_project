using ArgParse

function create_arg_parse_settings()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--parent_dataset_name"
        help = "Name of the folder housing the datasets"
        arg_type = String
        default = DEFAULTS[:PARENT_DATASET_NAME]
        nargs = 'A'
        "--n_bees"
        help = "The number of neural networks in the hive"
        arg_type = UInt16
        default = DEFAULTS[:N_BEES]
        nargs = 'A'
        "--n_epochs"
        help = "Number of epochs to train"
        arg_type = UInt16
        default = DEFAULTS[:N_EPOCHS]
        nargs = 'A'
        "--n_steps_per_epoch"
        help = "Number of actions every neural network will take on average in one epoch"
        arg_type = UInt16
        default = DEFAULTS[:N_STEPS_PER_EPOCH]
        nargs = 'A'
        "--learning_rate"
        help = "Learning rate for the update step of the neural networks"
        arg_type = Float16
        default = DEFAULTS[:LEARNING_RATE]
        nargs = 'A'
        "--punish_rate"
        help = "Prefactor for the negative update step of the dominated neural network"
        arg_type = Float32
        default = DEFAULTS[:PUNISH_RATE]
        nargs = 'A'
        "--random_seed"
        help = "The integer to set the seed for Random.seed!()"
        arg_type = Int
        default = DEFAULTS[:RANDOM_SEED]
        nargs = 'A'
        "--lambda_train"
        help = "The rate at which each individual neural networks will train"
        arg_type = Float16
        default = DEFAULTS[:LAMBDA_TRAIN]
        nargs = 'A'
        "--lambda_interact"
        help = "The exponent prefactor of the inverse sigmoid function of the interaction rate"
        arg_type = Float16
        default = DEFAULTS[:LAMBDA_INTERACT]
        nargs = 'A'
        "--accuracy_sigma"
        help = "The standard deviation of the Gaussian noise added to the accuracy of the neural networks"
        arg_type = Float16
        default = DEFAULTS[:ACCURACY_SIGMA]
        nargs = 'A'
    end
    
    return s

end 
