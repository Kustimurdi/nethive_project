#!/usr/bin/env bash
set -euo pipefail
echo "script started"

#parent_dataset_name="custom_classification_training_propensity_sweep"
parent_dataset_name="custom_classification_testing"

task="custom_classification"
qg_method="accuracy"
n_epochs=1
n_steps_per_epoch=5
n_bees=100

#learning_rates=(0.000001 0.00001 0.0001)
learning_rates=(0.001)
punish_rates=(0.001)
#training_propensity=(0.01 0.1 1 10 100)
training_propensity=(1)
#lambda_interacts=(1)
lambda_interacts=(5 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95)
#random_seeds=(1)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

features_dimensions=(10)
n_classes=(5)
n_per_class_train=(100)
n_per_class_test=(100)
sampling_gaussian_sigmas=(1.0)
class_center_radii=(5.0)

for lr in "${learning_rates[@]}"; do
    for pr in "${punish_rates[@]}"; do
        for tp in "${training_propensity[@]}"; do
            for li in "${lambda_interacts[@]}"; do
                for rs in "${random_seeds[@]}"; do
                    for fd in "${features_dimensions[@]}"; do
                        for nc in "${n_classes[@]}"; do
                            for npctrain in "${n_per_class_train[@]}"; do
                                for npctest in "${n_per_class_train[@]}"; do
                                    for sgs in "${sampling_gaussian_sigmas[@]}"; do
                                        for ccr in "${class_center_radii[@]}"; do
                                            #echo "Submitting: lr=$lr, pr=$pr, lt=$lt, li=$li, as=$as, seed=$seed"
                                            sbatch ./run_sbatch.slurm \
                                                --task=$task \
                                                --queen_gene_method=$qg_method \
                                                --parent_dataset_name=$parent_dataset_name \
                                                --n_bees=$n_bees \
                                                --n_epochs=$n_epochs \
                                                --n_steps_per_epoch=$n_steps_per_epoch \
                                                --learning_rate=$lr \
                                                --punish_rate=$pr \
                                                --training_propensity=$tp \
                                                --lambda_interact=$li \
                                                --random_seed=$rs \
                                                --features_dimension=$fd \
                                                --n_classes=$nc \
                                                --n_per_class_train=$npctrain \
                                                --n_per_class_test=$npctest \
                                                --class_center_radius=$ccr \
                                                --sampling_gauss_sigma=$sgs
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done