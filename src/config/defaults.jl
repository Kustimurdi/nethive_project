const DEFAULTS = Dict(
    :PARENT_DATASET_NAME => "default",
    :TASK_TYPE => :regression,
    :QUEEN_GENE_METHOD => :accuracy_based,
    :N_BEES => UInt16(3),
    :N_EPOCHS => UInt16(100),
    :N_STEPS_PER_EPOCH => 1,
    :LEARNING_RATE => Float16(0.0003),
    :PUNISH_RATE => Float32(0.0000001),
    :LAMBDA_TRAIN => Float16(0.05),
    :LAMBDA_INTERACT => Float16(5),
    :ACCURACY_SIGMA => Float16(1),
    :RANDOM_SEED => 1,
    :TRAINSET_SIZE => 10000,
    :TESTSET_SIZE => 1000,

    #Regression defaults
    :REGRESSION_N_PEAKS => 5,
    :REGRESSION_WHICH_PEAKS => 1
)
