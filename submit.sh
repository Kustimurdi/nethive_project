#!/usr/bin/env bash

sbatch ./run_sbatch.slurm --n_bees=3 --n_epochs=7 --parent_dataset_name="gillespie_alg_test_model_4" --n_steps_per_epoch=3