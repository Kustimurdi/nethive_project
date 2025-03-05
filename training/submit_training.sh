#!/usr/bin/env bash

sbatch /scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_project/training/run_training.slurm --n_bees=1 --n_epochs=5 --parent_dataset_name="training_neural_network_model_5" --learning_rate=0.0001
