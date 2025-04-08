
library(data.table)
library(ggplot2)
library(bektas.utils.r)

source("./methods.R")

data_path <- "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/training_neural_network_model_4/D250227T170512747I2"

accuracy_data <- read.csv(paste(data_path, "accuracy_history.csv", sep=''))

print(accuracy_data)

gg2 <- ggplot(accuracy_data, aes(x = epoch_index, y = accuracy, color = bee_id, group = bee_id)) +
    geom_line() +
    geom_point() +
    labs(title = "Accuracy Evolution",
        x = "Epochs",
        y = "Accuracy") +
    theme_minimal()

#cggsave(paste(data_path, "accuracy", sep=''), device = 'pdf', height = fig.height, width = fig.width,  plot = gg)



print("geeks")
