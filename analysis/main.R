#setwd("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project")
#renv::activate()

library(data.table)
library(ggplot2)

source("./methods.R")

data_path <- "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/raw/default/D250213T175611714I4"

accuracy_data <- read.csv(paste(data_path, "accuracy_history_epoch_3.csv"))

print(accuracy_data)


print("geeks")


args <- commandArgs(trailingOnly = TRUE)

default_dataset_name <- "D250213T175611714I4"
default_parent_dataset_name <- "default"

DATASET_NAME <- ifelse("--dataset" %in% args, args[which(args == "--dataset")+1], default_dataset_name)
PARENT_DATASET_NAME <- ifelse("--parent_dataset" %in% args, args[which(args == "--parent_dataset")+1], default_parent_dataset_name)
DATA_PATH <- paste("~/Code/nikolas/rulands.nikolas.smarticles/raw/", PARENT_DATASET_NAME, "/", DATASET_NAME, "/", sep='')
if (file.exists(paste(DATA_PATH, "parameters.csv", sep = ''))) {
    dt_parameters <- fread(paste(DATA_PATH, "parameters.csv", sep = ''))
} else {
    warning(sprintf("parameters.csv does not exist in %s", DATA_PATH))
}
sprintf("Path to dataset: %s", DATA_PATH)
print(dt_parameters)






filename <- "/scratch/n/N.Pfaffenzeller/"
data <- read.csv("")



data(mtcars)

gg <- ggplot(data=mtcars, aes(x=mpg, y=wt)) +
    geom_point() +
    labs(title = "Scatter Plot",
        x = "MPG",
        y = "Weight") +
    theme_minimal()

