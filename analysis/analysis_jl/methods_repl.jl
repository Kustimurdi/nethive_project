
"""
---------------------------------------
these functions are for inspection during repl sessions
---------------------------------------
"""

function plot_hive_predictions(h::Hive, dataset; title="predictions of the Hive", xlabel="x", ylabel="model(x)", markersize=4)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    scatter_plots = []
    labels = []
    for bee in h.bee_list
        predictions = bee.brain(dataset)
        scatter_plot = Makie.scatter!(ax, vec(dataset), vec(predictions), markersize=markersize)
        push!(scatter_plots, scatter_plot)
        push!(labels, "bee_$(bee.id)")
    end
    Legend(fig[1,2], scatter_plots, labels, "Bees")
    fig
end

function plot_bee_prediction(bee_brain, dataset; title="predictions of bee", xlabel="x", ylabel="model(x)", markersize=4)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    predictions = bee_brain(dataset)
    scatter_plot = Makie.scatter!(ax, vec(dataset), vec(predictions), markersize=markersize)
    fig
end

function plot_hive_history(dataset_matrix; title="Hive history", xlabel="epoch", ylabel="history", markersize=4)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)
    scatter_plots = []
    labels = []
    epochs = collect(1:size(dataset_matrix)[2])
    for row in 1:size(dataset_matrix)[1]
        scatter_plot = Makie.scatter!(ax, epochs, vec(dataset_matrix[row, :]), markersize=markersize)
        push!(scatter_plots, scatter_plot)
        push!(labels, "bee_$(row)")
    end
    Legend(fig[1,2], scatter_plots, labels, "Bees")
    fig
end

function plot_bee_history(dataset_matrix, bee_id; title="Hive history", xlabel="epoch", ylabel="history", markersize=4)
    fig = Figure()
    ax = Makie.Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)
    epochs = collect(1:size(dataset_matrix)[2])
    scatter_plot = Makie.scatter!(ax, epochs, vec(dataset_matrix[bee_id, :]), markersize=markersize)
    fig
end
