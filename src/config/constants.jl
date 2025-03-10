# Parse arguments
s = create_arg_parse_settings()
parsed_args = parse_args(ARGS, s)


# Define constants
const DATASET_NAME::String = string(Dates.format(now(), "DyymmddTHHMMSSss"), "I", rand(1:9, 1)[1])
const PARENT_DATASET_NAME::String = haskey(parsed_args, "parent_dataset_name") ? parsed_args["parent_dataset_name"] : DEFAULTS[:PARENT_DATASET_NAME]
const N_BEES::UInt16 = haskey(parsed_args, "n_bees") ? UInt16(parsed_args["n_bees"]) : DEFAULTS[:N_BEES]
const N_EPOCHS::UInt16 = haskey(parsed_args, "n_epochs") ? UInt16(parsed_args["n_epochs"]) : DEFAULTS[:N_EPOCHS]
const N_STEPS_PER_EPOCH::UInt16 = haskey(parsed_args, "n_steps_per_epoch") ? UInt16(parsed_args["n_steps_per_epoch"]) : DEFAULTS[:N_STESP_PER_EPOCH]
const LEARNING_RATE::Float16 = haskey(parsed_args, "learning_rate") ? Float16(parsed_args["learning_rate"]) : DEFAULTS[:LEARNING_RATE] 
const RANDOM_SEED::Float16 = haskey(parsed_args, "random_seed") ? Float16(parsed_args["random_seed"]) : DEFAULTS[:RANDOM_SEED] 
#const INPUT_SIZE::Vector{UInt16} = haskey(parsed_args, "input_size") ? parsed_args["input_size"] : DEFAULTS[:INPUT_SIZE]
#const OUTPUT_SIZE::UInt16 = haskey(parsed_args, "output_size") ? parsed_args["output_size"] : DEFAULTS[:OUTPUT_SIZE]


const RAW_PATH::String = string("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/", PARENT_DATASET_NAME, "/", DATASET_NAME)
const RAW_NET_PATH::String = string("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/", PARENT_DATASET_NAME, "/", DATASET_NAME, "/net/")


const GIT_COMMIT::String = "88a0d8f8febd806081b5429a22f91a65aa7dc52b"
const GIT_BRANCH::String = "Main"
