#!/usr/bin/env julia

import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")

using Printf
using Dates
using CSV
using DataFrames

"""
Functions for tracking parameter combinations in nethive sweeps.
Helps identify completed runs and generate only missing parameter files.
"""

"""
Scan all completed runs in a dataset and extract their parameter combinations.
Returns a DataFrame with all parameter combinations that have been completed.
"""
function scan_completed_runs(dataset_name::String)
    data_dir = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data"
    dataset_path = joinpath(data_dir, dataset_name)
    
    if !isdir(dataset_path)
        println("⚠️  Dataset directory not found: $dataset_path")
        return DataFrame()
    end
    
    completed_params = DataFrame()
    run_dirs = filter(d -> isdir(joinpath(dataset_path, d)), readdir(dataset_path))
    
    println("📊 Scanning $(length(run_dirs)) run directories...")
    
    for run_dir in run_dirs
        param_file = joinpath(dataset_path, run_dir, "parameters.csv")
        
        if isfile(param_file)
            try
                params = CSV.read(param_file, DataFrame)
                # Add run directory info
                params.run_directory = [run_dir]
                params.completed_at = [stat(param_file).mtime]
                
                if nrow(completed_params) == 0
                    completed_params = params
                else
                    completed_params = vcat(completed_params, params)
                end
            catch e
                println("⚠️  Could not read parameters from $param_file: $e")
            end
        else
            println("⚠️  No parameters.csv found in $run_dir")
        end
    end
    
    println("✅ Found $(nrow(completed_params)) completed runs")
    return completed_params
end

"""
Generate all possible parameter combinations based on the provided parameter ranges.
Returns a DataFrame with all combinations.
"""
function generate_all_combinations(;
    learning_rates = [0.001],
    punish_rates = [0.001],
    training_propensities = [1],
    lambda_interacts = [5],
    random_seeds = 1:5,
    features_dimensions = [10],
    n_classes = [5],
    n_per_class_train = [100],
    n_per_class_test = [100],
    sampling_gaussian_sigmas = [1.0],
    class_center_radii = [5.0],
    # Fixed parameters
    task_type = "custom_classification",
    qg_method = "accuracy",
    parent_dataset_name = "custom_classification_fallback",
    n_bees = [100],
    n_epochs = [1],
    n_steps_per_epoch = [5]
)
    
    combinations = DataFrame()
    
    for lr in learning_rates, pr in punish_rates, tp in training_propensities,
        li in lambda_interacts, rs in random_seeds, fd in features_dimensions,
        nc in n_classes, npctrain in n_per_class_train, npctest in n_per_class_test,
        sgs in sampling_gaussian_sigmas, ccr in class_center_radii, ne in n_epochs, 
        nsp in n_steps_per_epoch, nb in n_bees
        
        row = DataFrame(
            task_type = [task_type],
            queen_gene_method = [qg_method],
            parent_dataset_name = [parent_dataset_name],
            n_bees = [nb],
            n_epochs = [ne],
            n_steps_per_epoch = [nsp],
            learning_rate = [lr],
            punish_rate = [pr],
            training_propensity = [tp],
            lambda_interact = [li],
            random_seed = [rs],
            features_dimension = [fd],
            n_classes = [nc],
            n_per_class_train = [npctrain],
            n_per_class_test = [npctest],
            class_center_radius = [ccr],
            sampling_gauss_sigma = [sgs]
        )
        
        if nrow(combinations) == 0
            combinations = row
        else
            combinations = vcat(combinations, row)
        end
    end
    
    return combinations
end

"""
Find missing parameter combinations by comparing all possible combinations with completed ones.
"""
function find_missing_combinations(dataset_name::String; kwargs...)
    completed = scan_completed_runs(dataset_name)
    all_combinations = generate_all_combinations(; kwargs...)
    println("completed: $(nrow(completed)), all combinations: $(nrow(all_combinations))")
    
    if nrow(completed) == 0
        println("📝 No completed runs found. All $(nrow(all_combinations)) combinations are missing.")
        return all_combinations
    end
    
    # Define columns to compare (exclude run metadata)
    param_columns = setdiff(names(all_combinations), ["run_directory", "completed_at"])
    completed_columns = names(completed)
    # Only compare columns present in both DataFrames
    compare_columns = intersect(param_columns, completed_columns)
    missing_in_completed = setdiff(param_columns, completed_columns)
    extra_in_completed = setdiff(completed_columns, param_columns)
    if !isempty(missing_in_completed)
        println("[WARN] Columns missing in completed runs: ", missing_in_completed)
    end
    if !isempty(extra_in_completed)
        println("[INFO] Extra columns in completed runs: ", extra_in_completed)
    end
    println("[DEBUG] Using columns for comparison: ", compare_columns)
    # Align DataFrames to comparison columns
    completed_cmp = completed[:, compare_columns]
    all_combinations_cmp = all_combinations[:, compare_columns]
    # Normalize types for comparison
    for col in compare_columns
        if eltype(completed_cmp[!, col]) <: Number && eltype(all_combinations_cmp[!, col]) <: Number
            completed_cmp[!, col] = Float64.(completed_cmp[!, col])
            all_combinations_cmp[!, col] = Float64.(all_combinations_cmp[!, col])
        end
        if eltype(completed_cmp[!, col]) <: AbstractString || eltype(all_combinations_cmp[!, col]) <: AbstractString
            completed_cmp[!, col] = string.(completed_cmp[!, col])
            all_combinations_cmp[!, col] = string.(all_combinations_cmp[!, col])
        end
    end
    println("[DEBUG] Completed types: ", eltype.(eachcol(completed_cmp)))
    println("[DEBUG] All combinations types: ", eltype.(eachcol(all_combinations_cmp)))
    println("[DEBUG] Sample completed row: ", completed_cmp[1, :])
    println("[DEBUG] Sample all_combinations row: ", all_combinations_cmp[1, :])
    # Find missing combinations
    missing_combinations = DataFrame()
    for (i, row) in enumerate(eachrow(all_combinations_cmp))
        is_completed = false
        for comp_row in eachrow(completed_cmp)
            match = true
            for col in compare_columns
                if haskey(comp_row, col) && row[col] != comp_row[col]
                    match = false
                    break
                end
            end
            if match
                is_completed = true
                break
            end
        end
        if !is_completed
            # Add the full row from all_combinations (not just compare_columns)
            if nrow(missing_combinations) == 0
                missing_combinations = DataFrame(all_combinations[i, :])
            else
                missing_combinations = vcat(missing_combinations, DataFrame(all_combinations[i, :]))
            end
        end
    end
    println("📊 Summary:")
    println("   Total combinations: $(nrow(all_combinations))")
    println("   Completed: $(nrow(completed))")
    println("   Missing: $(nrow(missing_combinations))")
    return missing_combinations
end

"""
Generate parameter files only for missing combinations.
"""
function generate_missing_args_files(dataset_name::String, output_dir_name::String; kwargs...)
    missing = find_missing_combinations(dataset_name; kwargs...)
    
    if nrow(missing) == 0
        println("✅ All parameter combinations have been completed!")
        return 0
    end
    
    # Create output directory
    base_dir = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params"
    full_path = joinpath(base_dir, output_dir_name)
    isdir(full_path) || mkpath(full_path)
    
    # Clear existing files in the directory
    existing_files = filter(f -> endswith(f, ".txt"), readdir(full_path))
    for file in existing_files
        rm(joinpath(full_path, file))
    end
    
    counter = 1
    
    for row in eachrow(missing)
        arg_lines = [
            "--task_type=$(row.task_type)",
            "--queen_gene_method=$(row.queen_gene_method)",
            "--parent_dataset_name=$(row.parent_dataset_name)",
            "--n_bees=$(row.n_bees)",
            "--n_epochs=$(row.n_epochs)",
            "--n_steps_per_epoch=$(row.n_steps_per_epoch)",
            "--learning_rate=$(row.learning_rate)",
            "--punish_rate=$(row.punish_rate)",
            "--training_propensity=$(row.training_propensity)",
            "--lambda_interact=$(row.lambda_interact)",
            "--random_seed=$(row.random_seed)",
            "--features_dimension=$(row.features_dimension)",
            "--n_classes=$(row.n_classes)",
            "--n_per_class_train=$(row.n_per_class_train)",
            "--n_per_class_test=$(row.n_per_class_test)",
            "--class_center_radius=$(row.class_center_radius)",
            "--sampling_gauss_sigma=$(row.sampling_gauss_sigma)"
        ]
        
        filename = joinpath(full_path, @sprintf("args_%03d.txt", counter))
        write(filename, join(arg_lines, "\n"))
        counter += 1
    end
    
    println("✅ Generated $(counter - 1) parameter files for missing combinations in $full_path")
    return counter - 1
end

"""
Save a summary of completed and missing combinations to CSV files.
"""
function save_tracking_summary(dataset_name::String, output_dir::String; kwargs...)
    completed = scan_completed_runs(dataset_name)
    missing = find_missing_combinations(dataset_name; kwargs...)
    
    # Create output directory
    isdir(output_dir) || mkpath(output_dir)
    
    # Save completed runs
    if nrow(completed) > 0
        completed_file = joinpath(output_dir, "completed_runs.csv")
        CSV.write(completed_file, completed)
        println("💾 Saved completed runs summary to: $completed_file")
    end
    
    # Save missing combinations
    if nrow(missing) > 0
        missing_file = joinpath(output_dir, "missing_combinations.csv")
        CSV.write(missing_file, missing)
        println("💾 Saved missing combinations to: $missing_file")
    end
    
    # Save overall summary
    summary_data = DataFrame(
        metric = ["total_combinations", "completed", "missing"],
        count = [nrow(completed) + nrow(missing), nrow(completed), nrow(missing)]
    )
    summary_file = joinpath(output_dir, "sweep_summary.csv")
    CSV.write(summary_file, summary_data)
    println("💾 Saved sweep summary to: $summary_file")
end

# Example usage functions
"""
Example: Generate args files for lambda_interact sweep with finer granularity
"""
function generate_lambda_interact_sweep_incremental()
    # First sweep: 5:5:95 (already done)
    # Second sweep: fill in gaps with 2.5 steps
    lambda_values_fine = vcat(
        collect(2.5:5:97.5),  # Fill gaps: 2.5, 7.5, 12.5, ...
        collect(1:1:4),       # Very fine at low end: 1, 2, 3, 4
        collect(96:1:100)     # Very fine at high end: 96, 97, 98, 99, 100
    )
    
    return generate_missing_args_files(
        "custom_classification_testing",
        "lambda_interact_fine_sweep",
        lambda_interacts = lambda_values_fine
    )
end

"""
Example: Generate args files for learning rate sweep
"""
function generate_learning_rate_sweep()
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    return generate_missing_args_files(
        "custom_classification_testing", 
        "learning_rate_sweep",
        learning_rates = learning_rates
    )
end
