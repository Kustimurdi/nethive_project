const DEFAULTS = Dict(
    :parent_dataset_name => "default",
    :task_type => :regression,
    :queen_gene_method => :accuracy,
    :n_bees => UInt16(4),
    :n_epochs => UInt16(2000),
    :n_steps_per_epoch => 1,
    :learning_rate => Float32(0.00003),
    :punish_rate => Float32(0.0000001),
    :lambda_train => Float64(0.005),
    :lambda_interact => Float16(5),
    :accuracy_sigma => Float16(0.1),
    :random_seed => 1,
    :trainset_size => 10000,
    :testset_size => 1000,
    :initial_queen_gene => Float64(0.0),
    :save_nn_epochs => 0,

    #Regression defaults
    :regression_n_peaks => 5,
    :regression_which_peak => 1,

    #QueenGeneIncremental defaults
    :queen_gene_incremental_rate => 0.3,
    :queen_gene_decremental_rate => 0.3
)
