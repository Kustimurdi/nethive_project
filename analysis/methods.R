library(ggplot2)

plot_bees_evolution <- function(data, property_name, x_axis_name="Epochs", y_axis_name="", title_name="") { 
    if (!all(c("bee_id", "epoch_index", property_name) %in% colnames(data))) {
        stop(paste("The dataset must contain 'bee_id', 'epoch_index' and ", property_name, " columns.", sep=''))
    }
    
    p <- ggplot(data, aes_string(x = "epoch_index", y = property_name, color = "bee_id", group = "bee_id")) +
        geom_line() +
        geom_point() +
        labs(title = title_name,
            x = x_axis_name,
            y = y_axis_name) +
        theme_minimal()

    return(p)
}