"""
-------------------------------------------------
these functions are for inspection of saved data
-------------------------------------------------
"""
function read_parameters(df_parameters)
    return Dict(df_parameters.id .=> df_parameters.value)  
end

function read_parameters_from_file(file_path::String)
    df = CSV.read(file_path, DataFrame)
    return Dict(df.id .=> df.value)
end

function read_simulation_data(directory::String)
    files = Dict(
        "accuracy" => "accuracy_history.csv",
        "loss" => "loss_history.csv",
        "queen_gene" => "queen_genes_history.csv",
        "train_counts" => "train_history.csv",
        "dominant_counts" => "dominant_history.csv",
        "subdominant_counts" => "subdominant_history.csv",
        "parameters" => "parameters.csv"
    )

    data_dict = Dict(key => CSV.read(joinpath(directory, filename), DataFrame) 
                                  for (key, filename) in files if isfile(joinpath(directory, filename)))
    return data_dict
end

function plot_hive_data(df::DataFrame, column=:accuracy; which_bee=nothing, title="Hive history", xlabel="Epoch", ylabel="column_value", markersize=4)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)
    
    scatter_plots = []
    labels = []
    
    if isnothing(which_bee)
        bee_ids = unique(df[:, :bee_id])
    else
        bee_ids = which_bee
    end
    epochs = unique(df[:, :epoch_index])
    
    for bee_id in bee_ids
        # Filter rows corresponding to the current bee_id
        bee_data = df[df[:, :bee_id] .== bee_id, :]
        
        column_values = [bee_data[bee_data[:, :epoch_index] .== epoch, column][1] for epoch in epochs]
        
        scatter_plot = Makie.scatter!(ax, epochs, column_values, markersize=markersize)
        push!(scatter_plots, scatter_plot)
        push!(labels, "bee_$(bee_id)")
    end
    Legend(fig[1, 2], scatter_plots, labels, "Bees")
    return fig
end

function compute_dominance_ratio(sim_data)
    dominant_df = sim_data["dominant_counts"]
    subdominant_df = sim_data["subdominant_counts"]

    dominance_ratios_df = innerjoin(dominant_df, subdominant_df, on=:bee_id, makeunique=true)
    dominance_ratios_df[!, :dominance_ratio] = dominance_ratios_df[:, :n_dominant_interactions] ./ 
        (dominance_ratios_df[:, :n_dominant_interactions] .+ dominance_ratios_df[:, :n_subdominant_interactions])
    select!(dominance_ratios_df, Not([:epoch_index_1]))
    return dominance_ratios_df
end

function compute_ranking(sim_data; num_epochs_avg=5, w1=1.0, w2=1.0, w3=1.0, w4=1.0)
    accuracy_df = sim_data["accuracy"]
    dominant_df = sim_data["dominant_counts"]
    subdominant_df = sim_data["subdominant_counts"]
    
    final_accuracies = combine(
        groupby(accuracy_df, :bee_id),
        :accuracy => (x -> mean(last(x, num_epochs_avg))) => :final_accuracy
    )
    last_epochs = sort(unique(dominant_df.epoch_index), rev=true)[1:num_epochs_avg]

    dominant_df_last = filter(row -> row.epoch_index in last_epochs, dominant_df)
    subdominant_df_last = filter(row -> row.epoch_index in last_epochs, subdominant_df)

    merged_df = innerjoin(dominant_df_last, subdominant_df_last, on=[:bee_id, :epoch_index], 
                          makeunique=true)

    merged_df.dominance_ratio = merged_df.n_dominant_interactions ./ (merged_df.n_dominant_interactions .+ merged_df.n_subdominant_interactions)
    replace!(merged_df.dominance_ratio, NaN => 0.0)

    avg_ratios = combine(groupby(merged_df, :bee_id), :dominance_ratio => mean => :avg_dominance_ratio)

    max_dominance = maximum(avg_ratios[:, :avg_dominance_ratio])
    max_accuracy = maximum(final_accuracies[:, :final_accuracy])
    mean_accuracy_others = mean(setdiff(final_accuracies[:, :final_accuracy], max_accuracy))
    mean_dominance_others = mean(setdiff(merged_df[:, :dominance_ratio], max_dominance))
    
    rank_score = w1 * max_dominance + w2 * max_accuracy - w3 * mean_accuracy_others - w4 * mean_dominance_others
    #rank_score = w2 * max_accuracy - w3 * mean_accuracy_others
    
    println("max_dominance = $(max_dominance) : max_accuracy = $(max_accuracy) : mean_accuracy_others = $(mean_accuracy_others) : mean_dominance_others = $(mean_dominance_others) : rank = $(rank_score)")
    return rank_score, final_accuracies, avg_ratios
end

function analyse_all_simulations(base_directory::String, num_epochs_avg=5)
    simulation_dirs = filter(dir -> isdir(joinpath(base_directory, dir)), readdir(base_directory))
    results = []

    for dir in simulation_dirs
        sim_path = joinpath(base_directory, dir)
        sim_data = read_simulation_data(sim_path)

        required_keys = ["parameters", "accuracy", "loss", "dominant_counts", "subdominant_counts"]
        if all(k -> haskey(sim_data, k), required_keys)
            params = read_parameters_from_file(joinpath(sim_path, "parameters.csv"))
            accuracy_df = sim_data["accuracy"]
            loss_df = sim_data["loss"]
            train_count_df = get(sim_data, "train_count", nothing) #get() wird benutzt weil train_count fehlen darf (es ist nicht in den required_keys), in dem fall wird nothing returned
            dominant_df = sim_data["dominant_counts"]
            subdominant_df = sim_data["subdominant_counts"]

            if parse(Int, params["n_epochs"]) <= num_epochs_avg
                continue
            end

            rank_score, final_accuracies, final_dominance_ratios = compute_ranking(sim_data, num_epochs_avg=num_epochs_avg)

            # Store results
            push!(results, (dir, rank_score, params, final_accuracies, final_dominance_ratios))
        end
    end
    results = sort(results, by=x -> x[2], rev=true)

    return results
end



"""
dict_D250321T183138399I8 = honeyweb_data["D250321T183138399I8"]

sim_data = read_simulation_data("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/honeyweb/D250321T183138399I8")

compute_dominance_ratio(sim_data)
sim_data["dominant_counts"]

plot_hive_history_from_df(results[1][5], :dominance_ratio, which_bee=3)

rank, merge = compute_ranking(sim_data, num_epochs_avg=3)

data_dict = read_csv_files("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/default/D250321T152337869I2/")

data_acc = data_dict["accuracy_history.csv"]

plot_hive_history_from_df(data_acc)

groupby(sim_data["accuracy"], :bee_id)

final_accuracies = combine(
    groupby(sim_data["accuracy"], :bee_id),
    :accuracy => (x -> mean(last(x, 300))) => :final_accuracy
)

plot_hive_history_from_df(sim_data["accuracy"], :accuracy)


results = analyse_all_simulations("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/honeyweb")
"""


#data = read_simulation_data(joinpath("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/honeyweb", "D250321T182559673I4"))

#plot_hive_history_from_df(data["accuracy"])

