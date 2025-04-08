library(readr)
library(ggplot2)

plot_bees_evolution_old <- function(data, property_name, x_axis_name="Epochs", y_axis_name="", title_name="") { 
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

plot_bees_evolution <- function(data, property_name, x_axis_name="Epochs", y_axis_name="", title_name="") { 
    # Check that required columns exist
    if (!all(c("bee_id", "epoch_index", property_name) %in% colnames(data))) {
        stop(paste("The dataset must contain 'bee_id', 'epoch_index', and '", property_name, "' columns.", sep=''))
    }
    
    # Create the plot
    p <- ggplot(data, aes(x = epoch_index, y = .data[[property_name]], 
                          color = factor(bee_id), group = bee_id)) +
        geom_line() +
        geom_point() +
        labs(title = title_name,
             x = x_axis_name,
             y = y_axis_name,
             color = "Bee ID") +  # Label for the legend
        theme_minimal()

    return(p)
}

read_bee_history_data <- function(directory) {
    # Define the expected files
    expected_files <- c(
        "accuracy_history.csv", 
        "dom_interactions_history.csv", 
        "loss_history.csv", 
        "subdom_interactions_history.csv"
    )
    
    # Initialize an empty list to store datasets
    data_list <- list()

    # Loop over each expected file
    for (file_name in expected_files) {
        file_path <- file.path(directory, file_name)
        
        if (file.exists(file_path)) {
            data <- read_csv(file_path, col_types = cols())  # Read CSV file
            data_list[[file_name]] <- data  # Store in list with filename as key
        } else {
            warning(paste("File not found:", file_name))  # Warn if a file is missing
        }
    }

    return(data_list)  # Return named list of datasets
}
