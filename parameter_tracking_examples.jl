#!/usr/bin/env julia

"""
Example usage of the parameter tracking system for nethive sweeps.
This script demonstrates different ways to use the tracking functions.
"""

import Pkg
using Pkg
Pkg.activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/env_nethive/")

include("parameter_tracking.jl")

function example_usage()
    println("🔍 Parameter Tracking System Examples")
    println("=" ^ 50)
    
    # Example 1: Check what's already been completed
    println("\n1️⃣  Scanning completed runs...")
    completed = scan_completed_runs("custom_classification_testing")
    if nrow(completed) > 0
        println("📊 Completed parameter ranges:")
        for col in ["lambda_interact", "random_seed", "learning_rate"]
            if col in names(completed)
                values = sort(unique(completed[!, col]))
                println("   $col: $values")
            end
        end
    end
    
    # Example 2: Generate only missing combinations (basic sweep)
    println("\n2️⃣  Generating missing files for basic sweep...")
    n_basic = generate_missing_args_files(
        "custom_classification_testing",
        "sweep_basic_missing",
        lambda_interacts = 5:5:95,  # Your original sweep
        random_seeds = 1:5
    )
    
    # Example 3: Generate files for finer lambda_interact sweep
    println("\n3️⃣  Generating missing files for finer lambda sweep...")
    # Include both original values and new intermediate values
    lambda_fine = vcat(
        collect(5:5:95),     # Original values  
        collect(2.5:5:97.5), # Intermediate values: 2.5, 7.5, 12.5, ...
        collect(1:1:4)       # Very fine at low end
    )
    
    n_fine = generate_missing_args_files(
        "custom_classification_testing",
        "sweep_lambda_fine_missing", 
        lambda_interacts = sort(unique(lambda_fine)),
        random_seeds = 1:5
    )
    
    # Example 4: Generate files for learning rate sweep
    println("\n4️⃣  Generating missing files for learning rate sweep...")
    n_lr = generate_missing_args_files(
        "custom_classification_testing",
        "sweep_learning_rate_missing",
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01],
        lambda_interacts = [20, 50, 80],  # Test a few lambda values
        random_seeds = 1:3  # Fewer seeds for learning rate exploration
    )
    
    # Example 5: Save comprehensive tracking summary
    println("\n5️⃣  Saving tracking summary...")
    save_tracking_summary(
        "custom_classification_testing",
        "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params/tracking_summary",
        lambda_interacts = vcat(collect(5:5:95), collect(2.5:5:97.5)),
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01],
        random_seeds = 1:5
    )
    
    println("\n✅ Examples completed!")
    println("📂 Check the following directories for generated parameter files:")
    println("   • /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params/sweep_basic_missing")
    println("   • /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params/sweep_lambda_fine_missing") 
    println("   • /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params/sweep_learning_rate_missing")
    println("   • /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_params/tracking_summary")
end

# Uncomment to run examples
# example_usage()

# Quick functions for your common use cases:

function quick_lambda_fine_sweep()
    """Generate parameters for a finer lambda_interact sweep"""
    lambda_values = vcat(
        collect(5:5:95),      # Original coarse sweep
        collect(2.5:5:97.5),  # Fill in the gaps
        [1, 2, 3, 4, 96, 97, 98, 99, 100]  # Edge cases
    )
    
    return generate_missing_args_files(
        "custom_classification_testing",
        "lambda_fine_sweep",
        lambda_interacts = sort(unique(lambda_values)),
        random_seeds = 1:5
    )
end

function quick_learning_rate_sweep()
    """Generate parameters for learning rate exploration"""
    return generate_missing_args_files(
        "custom_classification_testing", 
        "learning_rate_sweep",
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        lambda_interacts = [10, 25, 50, 75, 90],  # Representative lambda values
        random_seeds = 1:3
    )
end

function check_progress()
    """Quick check of current progress"""
    completed = scan_completed_runs("custom_classification_testing")
    println("📊 Current progress:")
    println("   Total completed runs: $(nrow(completed))")
    
    if nrow(completed) > 0
        for param in ["lambda_interact", "learning_rate", "random_seed"]
            if param in names(completed)
                unique_vals = sort(unique(completed[!, param]))
                println("   $param: $(length(unique_vals)) unique values ($(minimum(unique_vals)) to $(maximum(unique_vals)))")
            end
        end
    end
end
