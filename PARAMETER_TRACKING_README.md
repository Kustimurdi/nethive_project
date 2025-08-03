# Parameter Tracking System for Nethive Sweeps

This system helps you track completed parameter combinations and generate only missing parameter files for incremental sweeps.

## Quick Start

### 1. Check Current Progress
```julia
include("parameter_tracking_examples.jl")
check_progress()
```

### 2. Generate Only Missing Parameter Files
Instead of regenerating all parameter files, generate only the missing ones:

```julia
# In generate_args_files.jl, uncomment this line:
generate_missing_args_files_smart()

# Or use directly:
include("parameter_tracking.jl")
generate_missing_args_files(
    "custom_classification_testing",  # Your dataset name
    "my_sweep_missing",              # Output directory
    lambda_interacts = 1:1:100,      # Your desired parameter range
    random_seeds = 1:10
)
```

### 3. Add Finer Granularity to Existing Sweep
```julia
# Original sweep: 5:5:95 (5, 10, 15, ..., 95)
# Add intermediate values: 2.5, 7.5, 12.5, ..., 97.5

lambda_fine = vcat(
    collect(5:5:95),     # Keep original values
    collect(2.5:5:97.5)  # Add intermediate values
)

generate_missing_args_files(
    "custom_classification_testing",
    "lambda_fine_sweep", 
    lambda_interacts = lambda_fine,
    random_seeds = 1:5
)
```

## Key Functions

### `scan_completed_runs(dataset_name)`
Scans your results directory and returns a DataFrame of all completed parameter combinations.

### `generate_missing_args_files(dataset_name, output_dir, kwargs...)`
Generates parameter files only for combinations that haven't been completed yet.

### `find_missing_combinations(dataset_name, kwargs...)`
Returns a DataFrame of parameter combinations that still need to be run.

### `save_tracking_summary(dataset_name, output_dir, kwargs...)`
Saves CSV files with:
- `completed_runs.csv`: All completed parameter combinations
- `missing_combinations.csv`: All missing parameter combinations  
- `sweep_summary.csv`: Summary statistics

## Workflow

1. **Initial sweep**: Generate all parameter files normally
2. **Run jobs**: Use your job array system
3. **Check progress**: Use `check_progress()` or `scan_completed_runs()`
4. **Identify gaps**: Use `find_missing_combinations()` 
5. **Fill gaps**: Use `generate_missing_args_files()` with refined parameters
6. **Repeat**: Continue iteratively refining your parameter space

## Example: Refining Lambda Interact Sweep

```julia
# 1. Check what's been completed
completed = scan_completed_runs("custom_classification_testing")

# 2. See current lambda_interact values
if "lambda_interact" in names(completed)
    completed_lambdas = sort(unique(completed.lambda_interact))
    println("Completed lambda values: $completed_lambdas")
end

# 3. Define finer sweep that includes gaps
lambda_fine = vcat(
    completed_lambdas,           # Keep what's done
    [2.5, 7.5, 12.5, 17.5],    # Add specific gaps
    collect(96:1:100)           # Add fine resolution at high end
)

# 4. Generate only missing combinations
generate_missing_args_files(
    "custom_classification_testing",
    "lambda_refined", 
    lambda_interacts = sort(unique(lambda_fine)),
    random_seeds = 1:5
)
```

## Directory Structure

```
nethive_params/
├── sweep_args/              # Original parameter files
├── lambda_fine_sweep/       # Refined lambda sweep parameters
├── learning_rate_sweep/     # Learning rate exploration parameters
└── tracking_summary/        # CSV files with progress tracking
    ├── completed_runs.csv
    ├── missing_combinations.csv
    └── sweep_summary.csv
```

## Benefits

- ⏱️  **Save time**: Only run missing combinations
- 🎯 **Targeted exploration**: Add fine-grained sweeps in interesting regions
- 📊 **Progress tracking**: Always know what's been completed
- 🔄 **Iterative refinement**: Easy to add more parameter values based on results
- 💾 **Audit trail**: CSV files track your complete parameter exploration history
