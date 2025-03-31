const DEFAULTS = Dict(
    :INPUT_SIZE => [UInt16(32), UInt16(32), UInt16(1)],
    :OUTPUT_SIZE => UInt16(10),
    :PARENT_DATASET_NAME => "default",
    :N_BEES => UInt16(3),
    :N_EPOCHS => UInt16(20),
    :N_STEPS_PER_EPOCH => 10,
    :LEARNING_RATE => Float16(0.0003),
    :PUNISH_RATE => Float32(0.0000001),
    :LAMBDA_TRAIN => Float16(0.05),
    :LAMBDA_INTERACT => Float16(5),
    :ACCURACY_ATOL => Float16(0.05),
    :RANDOM_SEED => 1
)

const MAPPING_OUPUT_RANGE = Dict(
    :MNIST => (1, 10),
    :CIFAR10 => (11, 20)
)

"""
export DEFAULT_INPUT_SIZE, DEFAULT_OUTPUT_SIZE, DEFAULT_N_BEES, DEFAULT_N_EPOCHS

const DEFAULT_INPUT_SIZE = UInt16[28, 28, 1]
const DEFAULT_OUTPUT_SIZE = UInt16(10)
const DEFAULT_N_BEES = UInt16(3)
const DEFAULT_N_EPOCHS = UInt16(5)

function _initialize_exports()
    for name in names(@__MODULE__, all=true)
        if startswith(string(name), "DEFAULT_")  # Only export constants starting with "DEFAULT_"
            export(Symbol(name))
        end
    end
end
"""


