import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")

using Printf
using Dates
#using FilesystemPaths  # or just use built-in `joinpath` if preferred

# Include the parameter tracking functions
include("parameter_tracking.jl")

function generate_args_files()
    # Option 1: Generate ALL parameter files (original behavior)
     generate_all_args_files()
    
    # Option 2: Generate only MISSING parameter files (recommended)
    # generate_missing_args_files_smart()
end

function generate_all_args_files()
    """Generate all parameter files regardless of what's already completed"""
# Create output directory
    base_dir = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params"
    ARG_DIR = "custom_classification_training_propensity_lambda_interact_heatmap_sweep_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))"
    full_path = joinpath(base_dir, ARG_DIR)
    isdir(full_path) || mkpath(full_path)

    # Parameters
    parent_dataset_name = "custom_classification_training_propensity_lambda_interact_heatmap_sweep_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))"
    task_type = "custom_classification"
    qg_method = "accuracy"
    n_epochs = 10000
    n_steps_per_epoch = 5
    n_bees = 100

    learning_rates = [0.001]
    punish_rates = [0.001]
    training_propensities = [0.01 0.1 1 10 100]
    #lambda_interacts = 5:5:95
    lambda_interacts = [5 10 15 20 25 30 35 40 45 55 60 65 70 75 80 85 90 95] # Original sweep
    random_seeds = 1:10

    features_dimensions = [10]
    n_classes = [5]
    n_per_class_train = [100]
    n_per_class_test = [100]
    sampling_gaussian_sigmas = [1.0]
    class_center_radii = [5.0]

    # Counter
    counter = 1

    # Generate combinations
    for lr in learning_rates, pr in punish_rates, tp in training_propensities,
        li in lambda_interacts, rs in random_seeds, fd in features_dimensions,
        nc in n_classes, npctrain in n_per_class_train, npctest in n_per_class_test,
        sgs in sampling_gaussian_sigmas, ccr in class_center_radii

        arg_lines = [
            "--task_type=$task_type",
            "--queen_gene_method=$qg_method",
            "--parent_dataset_name=$parent_dataset_name",
            "--n_bees=$n_bees",
            "--n_epochs=$n_epochs",
            "--n_steps_per_epoch=$n_steps_per_epoch",
            "--learning_rate=$lr",
            "--punish_rate=$pr",
            "--training_propensity=$tp",
            "--lambda_interact=$li",
            "--random_seed=$rs",
            "--features_dimension=$fd",
            "--n_classes=$nc",
            "--n_per_class_train=$npctrain",
            "--n_per_class_test=$npctest",
            "--class_center_radius=$ccr",
            "--sampling_gauss_sigma=$sgs"
        ]

        filename = joinpath(full_path, @sprintf("args_%03d.txt", counter))
        write(filename, join(arg_lines, "\n"))
        counter += 1
    end

    println("✅ Generated $(counter - 1) parameter files in $full_path")

end



function generate_missing_args_files_smart()
    """Generate only parameter files for combinations that haven't been completed yet"""
    
    # Define your parameter ranges here
    lambda_interacts = 5:5:95  # Your current sweep
    # For finer sweep, you could use: vcat(5:5:95, 2.5:5:97.5)
    
    # Generate missing parameter files
    n_generated = generate_missing_args_files(
        "custom_classification_lambda_interact_sweep",  # Your dataset name
        "custom_classification_lambda_interact_sweep_rest", # Output directory name
        learning_rates = [0.001],
        punish_rates = [0.001], 
        training_propensities = [1],
        lambda_interacts = lambda_interacts,
        random_seeds = 1:5,
        features_dimensions = [10],
        n_classes = [5],
        n_per_class_train = [100],
        n_per_class_test = [100],
        sampling_gaussian_sigmas = [1.0],
        class_center_radii = [5.0],
        n_epochs = [10000],
        n_bees = [100],
        n_steps_per_epoch = [5],
        task_type = "custom_classification",
        parent_dataset_name = "custom_classification_lambda_interact_sweep"
    )
    
    # Optionally save tracking summary
    """
    save_tracking_summary(
        "custom_classification_testing",
        "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params/tracking_summary"
    )
    """
    
    return n_generated
end

generate_args_files()