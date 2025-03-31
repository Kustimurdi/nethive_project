function plot_simulation_accuracies(accuracy_df::DataFrame, sim_name::String)
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title="Accuracy over Epochs - $(sim_name)", xlabel="Epoch", ylabel="Accuracy")

    for bee_id in unique(accuracy_df.bee_id)
        bee_data = accuracy_df[accuracy_df.bee_id .== bee_id, :]
        lines!(ax, bee_data.epoch_index, bee_data.accuracy, label="Bee $(bee_id)")
    end

    axislegend(ax)
    return fig
end

#prob unnecessary
function read_csv_files(directory::String)
    csv_files = filter(f -> endswith(f, ".csv"), readdir(directory; join=true))  # Get full paths of CSV files
    data_dict = Dict(basename(file) => CSV.read(file, DataFrame) for file in csv_files)  # Use only file names as keys
    return data_dict
end

#prob unnecessary
function load_all_simulations(base_directory::String)
    simulation_dirs = filter(dir -> isdir(joinpath(base_directory, dir)), readdir(base_directory))
    simulation_data = Dict(dir => read_csv_files(joinpath(base_directory, dir)) for dir in simulation_dirs)
    return simulation_data
end

function read_parameters(df_parameters)
    return Dict(df_parameters.id .=> df_parameters.value)  
end

function read_parameters_from_file(file_path::String)
    df = CSV.read(file_path, DataFrame)
    return Dict(df.id .=> df.value)
end


function analyse_train_interaction_ratio(sim_data)
    train_df = get(sim_data, "train_count", DataFrame(bee_id=[], epoch_index=[], train_count=[]))
    dominant_df = sim_data["dominant_counts"]
    subdominant_df = sim_data["subdominant_counts"]
    
    # Compute cumulative sums
    train_df[!, :cumulative_train] = cumsum(train_df[:, :train_count])
    dominant_df[!, :cumulative_dominant] = cumsum(dominant_df[:, :count_dominant])
    subdominant_df[!, :cumulative_subdominant] = cumsum(subdominant_df[:, :count_subdominant])
    
    # Compute total interactions and ratio
    total_interactions = dominant_df[:, :cumulative_dominant] #+ subdominant_df[:, :cumulative_subdominant]
    train_interaction_ratio = train_df[:, :cumulative_train] ./ total_interactions
    
    """
    # Plot the ratio evolution
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Epochs", ylabel="Train/Interaction Ratio", title="Training vs. Interaction Over Time")
    lines!(ax, train_df[:, :epoch_index], train_interaction_ratio)
    save("train_interaction_ratio.png", fig)
    """
end
