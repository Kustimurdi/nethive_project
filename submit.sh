#!/usr/bin/env bash

learning_rates=(0.0002 0.00025 0.0003 0.00035 0.0004)
punish_rates=(0.00000005 0.0000001 0.0000002 0.0000003)
lambda_trains=(0.002 0.003 0.004 0.006)
lambda_interacts=(5)
acc_atols=(0.05)

"""
learning_rates=(0.0004)
punish_rates=(0.0003)
lambda_trains=(0.0005 0.01 0.015)
lambda_interacts=(5)
acc_atols=(0.05)
"""


for lr in "${learning_rates[@]}"; do
    for pr in "${punish_rates[@]}"; do
        for lt in "${lambda_trains[@]}"; do
            for li in "${lambda_interacts[@]}"; do
                for aa in "${acc_atols[@]}"; do
                    sbatch ./run_sbatch.slurm --learning_rate=$lr --punish_rate=$pr --lambda_train=$lt --lambda_interact=$li --accuracy_atol=$aa --n_bees=5 --n_epochs=400 --parent_dataset_name="honeyweb3" --n_steps_per_epoch=10 --random_seed=1 --accuracy_atol=0.1
                done
            done
        done
    done
done