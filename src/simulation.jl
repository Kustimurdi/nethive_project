

function run_simulation(parsed_args::Dict{String, Any}; save_results::Bool = true, verbose::Bool = true, seed::Int = DEFAULTS["random_seed"])
    # Set the random seed for reproducibility
    Random.seed!(seed)
    
    # Logger setup
    logger = ConsoleLogger(stderr, Logging.Info)
    global_logger(logger)

    # Verbose logging setup
    if !verbose
        global_logger(ConsoleLogger(stderr, Logging.Warn))  # Only show warnings and errors
    end

    # Log start of simulation
    @info "Starting the simulation with parameters: $(parsed_args)"

    try
        # Hive configuration and paths
        @info "Creating HiveConfig..."
        hive_config = create_hive_config(parsed_args)
        
        @info "Setting up paths..."
        hive_paths = create_hive_paths(hive_config)

        @info "Building hive..."
        hive = Hive(hive_config)

        # Gillespie simulation (assuming it may take time)
        @info "Running Gillespie simulation..."
        gillespie_simulation(hive)

        # Saving results (only if requested)
        if save_results
            @info "Saving simulation results..."
            save_params(hive_config, hive_paths.raw_path)
            save_data(hive_paths.raw_path, hive)
            @info "Simulation data saved."
        end

        @info "Simulation completed successfully."
    catch e
        @error "An error occurred during the simulation: $e"
        rethrow(e)  # Re-throw the error to allow for debugging
    end
    
    return hive
end
