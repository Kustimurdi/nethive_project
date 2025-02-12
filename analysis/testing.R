
library(data.table)
library(ggplot2)
library(bektas.utils.r)

source("./methods.R")

data_path <- "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/default/D250210T201831023I2/"

accuracy_data <- read.csv(paste(data_path, "accuracy_history_epoch_3.csv", sep=''))

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
