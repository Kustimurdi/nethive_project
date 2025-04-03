# Parse arguments
s = create_arg_parse_settings()
parsed_args = parse_args(ARGS, s)

println("learning rate : $(parsed_args["learning_rate"])")
println("punish rate : $(parsed_args["punish_rate"])")

# Define constants
const DATASET_NAME::String = string(Dates.format(now(), "DyymmddTHHMMSSss"), "I", rand(1:9, 1)[1])
const PARENT_DATASET_NAME::String = haskey(parsed_args, "parent_dataset_name") ? parsed_args["parent_dataset_name"] : DEFAULTS[:PARENT_DATASET_NAME]
const N_BEES::UInt16 = haskey(parsed_args, "n_bees") ? UInt16(parsed_args["n_bees"]) : DEFAULTS[:N_BEES]
const N_EPOCHS::UInt16 = haskey(parsed_args, "n_epochs") ? UInt16(parsed_args["n_epochs"]) : DEFAULTS[:N_EPOCHS]
const N_STEPS_PER_EPOCH::UInt16 = haskey(parsed_args, "n_steps_per_epoch") ? UInt16(parsed_args["n_steps_per_epoch"]) : DEFAULTS[:N_STESP_PER_EPOCH]
const LEARNING_RATE::Float16 = haskey(parsed_args, "learning_rate") ? Float16(parsed_args["learning_rate"]) : DEFAULTS[:LEARNING_RATE] 
const PUNISH_RATE::Float32 = haskey(parsed_args, "punish_rate") ? Float32(parsed_args["punish_rate"]) : DEFAULTS[:PUNISH_RATE] 
const RANDOM_SEED::Float16 = haskey(parsed_args, "random_seed") ? Float16(parsed_args["random_seed"]) : DEFAULTS[:RANDOM_SEED] 
const ACCURACY_SIGMA::Float16 = haskey(parsed_args, "accuracy_sigma") ? Float16(parsed_args["accuracy_sigma"]) : DEFAULTS[:ACCURACY_SIGMA] 
const LAMBDA_TRAIN::Float16 = haskey(parsed_args, "lambda_train") ? Float16(parsed_args["lambda_train"]) : DEFAULTS[:LAMBDA_TRAIN] 
const LAMBDA_INTERACT::Float16 = haskey(parsed_args, "lambda_interact") ? Float16(parsed_args["lambda_interact"]) : DEFAULTS[:LAMBDA_INTERACT] 


const RAW_PATH::String = string("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/", PARENT_DATASET_NAME, "/", DATASET_NAME)
const RAW_NET_PATH::String = string("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/", PARENT_DATASET_NAME, "/", DATASET_NAME, "/net/")
const RAW_TASKDATA_PATH::String = string("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/", PARENT_DATASET_NAME, "/", DATASET_NAME, "/taskdata/")



const GIT_COMMIT::String = "ef03ab53396173c305026e21174f1e4233847fdd"
const GIT_BRANCH::String = "Main"
