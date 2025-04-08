library(data.table)
library(ggplot2)

source("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/analysis/methods.R")

current_dir <- getwd()

data_list <- read_bee_history_data(current_dir)

df_accuracy <- data_list[["accuracy_history.csv"]]

plot_bees_evolution(df_accuracy, "accuracy")

#/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/analysis/run_analysis.R