function create_arg_parse_settings(defaults::Dict{Symbol, Any} = DEFAULTS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--parent_dataset_name"
        help = "Name of the folder housing the datasets"
        arg_type = String
        default = defaults[:PARENT_DATASET_NAME]
        nargs = 'A'
        "--task_type"
        help = "The type of task to be performed"
        arg_type = Symbol
        default = defaults[:TASK_TYPE]
        nargs = 'A'
        "--queen_gene_method"
        help = "The method to select the queen gene"
        arg_type = Symbol
        default = defaults[:QUEEN_GENE_METHOD]
        nargs = 'A'
        "--n_bees"
        help = "The number of neural networks in the hive"
        arg_type = UInt16
        default = defaults[:N_BEES]
        nargs = 'A'
        "--n_epochs"
        help = "Number of epochs to train"
        arg_type = UInt16
        default = defaults[:N_EPOCHS]
        nargs = 'A'
        "--n_steps_per_epoch"
        help = "Number of actions every neural network will take on average in one epoch"
        arg_type = UInt16
        default = defaults[:N_STEPS_PER_EPOCH]
        nargs = 'A'
        "--learning_rate"
        help = "Learning rate for the update step of the neural networks"
        arg_type = Float16
        default = defaults[:LEARNING_RATE]
        nargs = 'A'
        "--punish_rate"
        help = "Prefactor for the negative update step of the dominated neural network"
        arg_type = Float32
        default = defaults[:PUNISH_RATE]
        nargs = 'A'
        "--lambda_train"
        help = "The rate at which each individual neural networks will train"
        arg_type = Float16
        default = defaults[:LAMBDA_TRAIN]
        nargs = 'A'
        "--lambda_interact"
        help = "The exponent prefactor of the inverse sigmoid function of the interaction rate"
        arg_type = Float16
        default = defaults[:LAMBDA_INTERACT]
        nargs = 'A'
        "--accuracy_sigma"
        help = "The standard deviation of the Gaussian noise added to the accuracy of the neural networks"
        arg_type = Float16
        default = defaults[:ACCURACY_SIGMA]
        nargs = 'A'
        "--random_seed"
        help = "The integer to set the seed for Random.seed!()"
        arg_type = Int
        default = defaults[:RANDOM_SEED]
        nargs = 'A'
        "--trainset_size"
        help = "The size of the training set"
        arg_type = Int
        default = defaults[:TRAINSET_SIZE]
        nargs = 'A'
        "--testset_size"
        help = "The size of the test set"
        arg_type = Int
        default = defaults[:TESTSET_SIZE]
        nargs = 'A'
        "--regression_n_peaks"
        help = "The number of sinus peaks in the regression task"
        arg_type = Int
        default = defaults[:REGRESSION_N_PEAKS]
        nargs = 'A'
        "--regression_which_peaks"
        help = "The index of the sinus peak in the regression task"
        arg_type = Int
        default = defaults[:REGRESSION_WHICH_PEAKS]
        nargs = 'A'
    end
    
    return s

end 
