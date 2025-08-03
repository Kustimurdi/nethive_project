function create_arg_parse_settings(defaults::Dict{Symbol, Any} = DEFAULTS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--args_file"
        help = "Path to a parameter file containing flags"
        arg_type = String
        default = ""
        nargs = 'A'
        "--parent_dataset_name"
        help = "Name of the folder housing the datasets"
        arg_type = String
        default = defaults[:parent_dataset_name]
        nargs = 'A'
        "--task_type"
        help = "The type of task to be performed"
        arg_type = Symbol
        default = defaults[:task_type]
        nargs = 'A'
        "--queen_gene_method"
        help = "The method to select the queen gene"
        arg_type = Symbol
        default = defaults[:queen_gene_method]
        nargs = 'A'
        "--n_bees"
        help = "The number of neural networks in the hive"
        arg_type = UInt16
        default = defaults[:n_bees]
        nargs = 'A'
        "--n_epochs"
        help = "Number of epochs to train"
        arg_type = UInt64
        default = defaults[:n_epochs]
        nargs = 'A'
        "--n_steps_per_epoch"
        help = "Number of actions every neural network will take on average in one epoch"
        arg_type = UInt16
        default = defaults[:n_steps_per_epoch]
        nargs = 'A'
        "--learning_rate"
        help = "Learning rate for the update step of the neural networks"
        arg_type = Float32
        default = defaults[:learning_rate]
        nargs = 'A'
        "--punish_rate"
        help = "Prefactor for the negative update step of the dominated neural network"
        arg_type = Float32
        default = defaults[:punish_rate]
        nargs = 'A'
        "--training_propensity"
        help = "The rate at which each individual neural networks will train"
        arg_type = Float64
        default = defaults[:training_propensity]
        nargs = 'A'
        "--lambda_interact"
        help = "The exponent prefactor of the inverse sigmoid function of the interaction rate"
        arg_type = Float16
        default = defaults[:lambda_interact]
        nargs = 'A'
        "--accuracy_sigma"
        help = "The standard deviation of the Gaussian noise added to the accuracy of the neural networks"
        arg_type = Float16
        default = defaults[:accuracy_sigma]
        nargs = 'A'
        "--random_seed"
        help = "The integer to set the seed for Random.seed!()"
        arg_type = Int
        default = defaults[:random_seed]
        nargs = 'A'
        "--trainset_size"
        help = "The size of the training set"
        arg_type = Int
        default = defaults[:trainset_size]
        nargs = 'A'
        "--testset_size"
        help = "The size of the test set"
        arg_type = Int
        default = defaults[:testset_size]
        nargs = 'A'
        "--regression_n_peaks"
        help = "The number of sinus peaks in the regression task"
        arg_type = Int
        default = defaults[:regression_n_peaks]
        nargs = 'A'
        "--regression_which_peak"
        help = "The index of the sinus peak in the regression task"
        arg_type = Int
        default = defaults[:regression_which_peak]
        nargs = 'A'
        "--initial_queen_gene"
        help = "The initial value of the queen gene"
        arg_type = Float64
        default = defaults[:initial_queen_gene]
        nargs = 'A'
        "--save_nn_epochs"
        help = "The epoch steps at which the neural networks get saved"
        arg_type = Int
        default = defaults[:save_nn_epochs]
        nargs = 'A'
        "--features_dimension"
        help = "The dimension of the vectors holding the features of the synthetic data for the custom classification task"
        arg_type = Int
        default = defaults[:features_dimension]
        nargs = 'A'
        "--n_classes"
        help = "The number of classes for the custom classification task"
        arg_type = Int
        default = defaults[:n_classes]
        nargs = 'A'
        "--n_per_class_train"
        help = "The number of samples per class for the custom classification task for the training set"
        arg_type = Int
        default = defaults[:n_per_class_train]
        nargs = 'A'
        "--n_per_class_test"
        help = "The number of samples per class for the custom classification task for the testing set"
        arg_type = Int
        default = defaults[:n_per_class_test]
        nargs = 'A'
        "--class_center_radius"
        help = "The scaling factor for the vectors holding the center of the classes for the custom classification task"
        arg_type = Float64
        default = defaults[:class_center_radius]
        nargs = 'A'
        "--sampling_gauss_sigma"
        help = "The sigma for the gaussian distribution from which the samples for the custom classification task will be drawn"
        arg_type = Float64
        default = defaults[:sampling_gauss_sigma]
        nargs = 'A'
    end
    
    return s

end 
